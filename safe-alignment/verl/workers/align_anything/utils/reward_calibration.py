# Copyright 2025 Rihong Qiu. All Rights Reserved.
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
# ==============================================================================

from __future__ import annotations

import json
import os
from typing import Tuple

import torch
from torch import Tensor


def estimate_alpha_beta(r_all: Tensor, p: float = 0.1) -> Tuple[float, float]:
    """Estimate calibration parameters (alpha, beta) from reward distribution.

    This function fits a simple location-scale transformation followed by a
    sigmoid non-linearity so that the calibrated score

        f(r) = sigmoid((r - alpha) / beta)

    approximately satisfies two quantile constraints

        f(q_lo) ~= p,   f(q_hi) ~= 1 - p,

    where ``q_lo`` and ``q_hi`` are the ``p`` and ``1 - p`` quantiles of the
    reward distribution ``r_all``.

    Args:
        r_all: A 1D or N-D tensor containing raw reward logits collected from
            the training or evaluation distribution.
        p: Lower tail probability used to define the calibration targets.
            Must be in the open interval (0, 0.5).

    Returns:
        A tuple ``(alpha, beta)`` of floats. ``alpha`` is a shift parameter and
        ``beta`` is a positive temperature parameter.
    """

    if not isinstance(r_all, torch.Tensor):
        raise TypeError("r_all must be a torch.Tensor.")

    if not (0.0 < p < 0.5):
        raise ValueError("p must be in the open interval (0, 0.5).")

    # Flatten to a 1D tensor for quantile computation.
    rewards = r_all.detach().float().view(-1)
    if rewards.numel() == 0:
        raise ValueError("r_all must contain at least one element.")

    # Compute lower and upper quantiles.
    q_lo = torch.quantile(rewards, p)
    q_hi = torch.quantile(rewards, 1.0 - p)

    # Symmetric logit targets for p and 1 - p.
    p_tensor = torch.tensor(float(p), dtype=rewards.dtype, device=rewards.device)
    l_lo = torch.log(p_tensor / (1.0 - p_tensor))
    l_hi = -l_lo  # logit(1 - p) = -logit(p)

    # Guard against degenerate cases where q_hi ~= q_lo.
    denom = (l_hi - l_lo).clamp_min(1e-8)
    if torch.isclose(q_hi, q_lo):
        beta = torch.tensor(1.0, dtype=rewards.dtype, device=rewards.device)
        alpha = q_lo
    else:
        beta = (q_hi - q_lo) / denom
        # Ensure beta is positive and not vanishingly small.
        beta = beta.clamp_min(1e-8)
        alpha = q_lo - beta * l_lo

    return float(alpha.item()), float(beta.item())


def calibrate_score(raw_reward: Tensor, alpha: float, beta: float) -> Tensor:
    """Apply shift + temperature calibration to raw reward logits.

    Given raw reward logits ``raw_reward``, this function applies the affine
    transformation ``(r - alpha) / beta`` followed by a sigmoid. The output is
    clamped to ``[0, 1]`` for numerical stability while preserving the input
    shape.

    Args:
        raw_reward: Tensor of raw reward logits of arbitrary shape.
        alpha: Shift parameter of the calibration.
        beta: Temperature parameter of the calibration. Should be positive.

    Returns:
        A tensor of the same shape as ``raw_reward`` with values in ``[0, 1]``.
    """

    if not isinstance(raw_reward, torch.Tensor):
        raise TypeError("raw_reward must be a torch.Tensor.")

    if beta == 0.0:
        raise ValueError("beta must be non-zero.")

    reward = raw_reward.detach().clone().float()
    z = (reward - float(alpha)) / float(beta)
    scores = torch.sigmoid(z)
    return scores.clamp_(0.0, 1.0)


def load_calibration(path: str) -> Tuple[float, float]:
    """Load calibration parameters from a JSON file.

    The JSON file is expected to contain a dictionary with the following keys::

        {"alpha": float, "beta": float}

    Args:
        path: Path to the JSON file.

    Returns:
        A tuple ``(alpha, beta)``.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "alpha" not in data or "beta" not in data:
        raise KeyError("Calibration file must contain 'alpha' and 'beta' keys.")

    alpha_val = float(data["alpha"])
    beta_val = float(data["beta"])
    return alpha_val, beta_val


def save_calibration(path: str, alpha: float, beta: float) -> None:
    """Save calibration parameters to a JSON file.

    The file will contain a minimal JSON object of the form::

        {"alpha": <float>, "beta": <float>}

    Parent directories are created if they do not exist.

    Args:
        path: Target path of the JSON file.
        alpha: Shift parameter to be saved.
        beta: Temperature parameter to be saved.
    """

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {"alpha": float(alpha), "beta": float(beta)}

    # Use json.dumps to keep control over encoding and separators.
    json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)
