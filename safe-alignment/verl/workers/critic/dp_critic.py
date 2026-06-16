# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available,get_device_id
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

        # 下面要加上gdpo_gae的方法
        # Multi-head value for GDPO_GAE: one independent Linear head per reward dimension.
        # The backbone (critic_module) is shared; only value_heads are dimension-specific.
        #
        # Configure via `critic.gdpo_gae_dimensions` – an explicit list of reward key names
        # (e.g. [format, correctness, integer]). The provided ordering is preserved and
        # must match the ordering assumed in compute_advantage (GDPO_GAE branch).
        raw_dims = self.config.model.get("gdpo_gae_dimensions", None)
        adv_estimator = self.config.get("adv_estimator", "")
        self.gdpo_gae_dimensions = list(raw_dims) if raw_dims and adv_estimator == "gdpo_gae" else []
        if self.gdpo_gae_dimensions:
            # Retrieve hidden_size by walking through possible wrappers (FSDP, etc.)
            def _get_hidden_size(module):
                if hasattr(module, "config") and hasattr(module.config, "hidden_size"):
                    return module.config.hidden_size
                if hasattr(module, "module"):
                    return _get_hidden_size(module.module)
                raise ValueError(f"[GDPO_GAE] Cannot determine hidden_size from critic_module. "
                               f"Module type: {type(module)}")

            hidden_size = _get_hidden_size(self.critic_module)
            n_heads = len(self.gdpo_gae_dimensions)
            self.value_heads = nn.ModuleList([nn.Linear(hidden_size, 1, bias=False) for _ in range(n_heads)])
            
            # Register value_heads parameters into the existing optimizer so they are
            # updated together with the backbone.
            self.critic_optimizer.add_param_group({"params": list(self.value_heads.parameters())})
            
            # Move to the same device / dtype as the backbone
            # Get dtype from critic_module to align with training parameters
            dtype = next(self.critic_module.parameters()).dtype
            self.value_heads.to(get_device_id()).to(dtype)
            
            # # Move to the same device / dtype as the backbone
            # # Only move to device if critic_module is not FSDP-wrapped yet
            # if not isinstance(self.critic_module, FSDP):
            #     # Get dtype from critic_module to avoid hardcoding
            #     dtype = next(self.critic_module.parameters()).dtype
            #     self.value_heads.to(get_device_id()).to(dtype)
            
            print(f"[GDPO_GAE] Initialized {n_heads} value heads for dims: {self.gdpo_gae_dimensions}")
        else:
            self.value_heads = None


    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )
        # Multi-head mode requires extracting the backbone's last hidden state.
        # It is incompatible with trl.AutoModelForCausalLMWithValueHead which owns its
        # own single value head; assert early to surface a clear error.
        is_multi_head = self.value_heads is not None
        if is_multi_head:
            assert not hasattr(self.critic_module, "v_head"), (
                "[GDPO_GAE] Multi-head critic is not compatible with trl.AutoModelForCausalLMWithValueHead. "
                "Please use a plain CausalLM as the backbone."
            )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                if is_multi_head:
                    # ---- Multi-head path (GDPO_GAE) ----
                    # Extract the last hidden state and project through each value head.
                    output = self.critic_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    # hidden_states[-1]: (1, total_nnz, H) → squeeze batch dim → (total_nnz, H)
                    last_hidden = output.hidden_states[-1].squeeze(0)
                    # Project each head: (total_nnz, 1) → squeeze → (total_nnz,); stack → (total_nnz, N)
                    values_rmpad = torch.stack(
                        [head(last_hidden).squeeze(-1) for head in self.value_heads], dim=-1
                    )  # (total_nnz, N)

                    # gather output if sp > 1
                    if self.ulysses_sequence_parallel_size > 1:
                        values_rmpad = gather_outputs_and_unpad(
                            values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                    # pad it back → (batch, seqlen, N), slice → (batch, response_length, N)
                    values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                    values = values[:, -response_length - 1 : -1]  # (B, response_length, N)
                else:
                    # ---- Single-head path (original) ----
                    output = self.critic_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                    )  # prevent model thinks we are generating

                    if hasattr(self.critic_module, "v_head"):
                        # For trl.AutoModelForCausalLMWithValueHead
                        values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                    else:
                        values_rmpad = output.logits
                        values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                    # gather output if sp > 1
                    if self.ulysses_sequence_parallel_size > 1:
                        values_rmpad = gather_outputs_and_unpad(
                            values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                    # pad it back
                    values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                    values = values[:, -response_length - 1 : -1]
            else:
                if is_multi_head:
                    # ---- Multi-head path (GDPO_GAE) ----
                    output = self.critic_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    last_hidden = output.hidden_states[-1]  # (B, S, H)
                    values = torch.stack(
                        [head(last_hidden).squeeze(-1) for head in self.value_heads], dim=-1
                    )  # (B, S, N)
                    values = values[:, -response_length - 1 : -1]  # (B, response_length, N)
                else:
                    # ---- Single-head path (original) ----
                    output = self.critic_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                    )  # prevent model thinks we are generating
                    if hasattr(self.critic_module, "v_head"):
                        # For trl.AutoModelForCausalLMWithValueHead
                        values = output[2]
                    else:
                        values = output.logits
                    values = values[:, -response_length - 1 : -1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        # multi-head modify  Also clip value_heads gradients (they are plain Linear layers, not wrapped in FSDP)
        if self.value_heads is not None:
            torch.nn.utils.clip_grad_norm_(self.value_heads.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                values = self._forward_micro_batch(model_inputs)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        response_mask = data.batch["response_mask"]
        # 这里对于多个head输出的值要对response_mask进行相同的扩展，以便正确地mask掉padding部分的值
        if values.dim() == 3:
            # Multi-head: values is (B, S, N), response_mask is (B, S) → broadcast over N
            values = values * response_mask.unsqueeze(-1)
        else:
            values = values * response_mask  # Only action tokens have values
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        # In multi-head (GDPO_GAE) mode the batch carries returns_per_dim (B, S, N) instead of
        # a single returns (B, S).  Detect this and choose the right select_keys.
        is_multi_head_update = "returns_per_dim" in data.batch
        if is_multi_head_update:
            select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns_per_dim"]
        else:
            select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    

                    vpreds = self._forward_micro_batch(model_inputs)
                    if is_multi_head_update:
                        # ---- Multi-head path (GDPO_GAE) ----
                        # returns_per_dim: (B, S, N), values: (B, S, N), vpreds: (B, S, N)
                        returns_per_dim = model_inputs["returns_per_dim"]
                        values_old = model_inputs["values"]
                        n_heads = returns_per_dim.shape[-1]
                        total_vf_loss = torch.zeros((), device=vpreds.device, dtype=vpreds.dtype)
                        total_vf_clipfrac = 0.0
                        for d in range(n_heads):
                            vf_loss_d, vf_clipfrac_d = core_algos.compute_value_loss(
                                vpreds=vpreds[:, :, d],
                                values=values_old[:, :, d],
                                returns=returns_per_dim[:, :, d],
                                response_mask=response_mask,
                                cliprange_value=self.config.cliprange_value,
                                loss_agg_mode=self.config.loss_agg_mode,
                            )
                            total_vf_loss = total_vf_loss + vf_loss_d
                            total_vf_clipfrac += vf_clipfrac_d.item()
                            # Record per-head loss and clipfrac in metrics
                            micro_batch_metrics[f"critic/vf_loss_head_{d}"] = vf_loss_d.detach().item()
                            micro_batch_metrics[f"critic/vf_clipfrac_head_{d}"] = vf_clipfrac_d.detach().item()
                        vf_loss = total_vf_loss / n_heads
                        vf_clipfrac_val = total_vf_clipfrac / n_heads

                    else:
                        # ---- Single-head path (original) ----
                        values = model_inputs["values"]
                        returns = model_inputs["returns"]
                        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                            cliprange_value=self.config.cliprange_value,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        vf_clipfrac_val = vf_clipfrac.detach().item()

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = vf_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation

                    loss.backward()

                    # For multi-head vpreds (B, S, N), average over heads before masked_mean
                    vpreds_for_mean = vpreds.mean(-1) if vpreds.dim() == 3 else vpreds
                    micro_batch_metrics.update(
                        {
                            "critic/vf_loss": vf_loss.detach().item(),
                            "critic/vf_clipfrac": vf_clipfrac_val,
                            "critic/vpred_mean": masked_mean(vpreds_for_mean, response_mask).detach().item(),
                        }
                    )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.critic_optimizer.zero_grad()
        return metrics
