#!/usr/bin/env python3
"""Parse TensorBoard event files and print validation mean@1 metrics."""
import argparse
import os

import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing import event_accumulator


TAGS = [
    "val-aux/Stanford Alpaca/reward_useful/mean@1",
    "val-aux/Stanford Alpaca/reward_harmless/mean@1",
    "val-core/Stanford Alpaca/reward/mean@1",
    "val-aux/Anthropic/hh-rlhf/reward_useful/mean@1",
    "val-aux/Anthropic/hh-rlhf/reward_harmless/mean@1",
    "val-aux/Anthropic/hh-rlhf/reward/mean@1",
    "val-aux/PKU-Alignment/PKU-SafeRLHF/reward_useful/mean@1",
    "val-aux/PKU-Alignment/PKU-SafeRLHF/reward_harmless/mean@1",
    "val-aux/PKU-Alignment/PKU-SafeRLHF/reward/mean@1",
]


def resolve_event_path(path: str) -> str:
    """Resolve a directory to the latest TensorBoard event file inside it."""
    if os.path.isdir(path):
        events = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.startswith("events.out.tfevents")
        ]
        if not events:
            raise FileNotFoundError(f"No TensorBoard event files found in {path}")
        return max(events, key=os.path.getmtime)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="TensorBoard event file or directory")
    parser.add_argument("--step", type=int, default=None, help="Step to read; defaults to latest for each tag")
    args = parser.parse_args()

    event_path = resolve_event_path(args.events)
    accumulator = event_accumulator.EventAccumulator(event_path)
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    print(f"events: {event_path}")
    rows = []
    for tag in TAGS:
        if tag not in scalar_tags:
            rows.append((tag, None, None))
            continue
        scalars = accumulator.Scalars(tag)
        if args.step is None:
            scalar = scalars[-1]
        else:
            matches = [item for item in scalars if item.step == args.step]
            scalar = matches[-1] if matches else None
        if scalar is None:
            rows.append((tag, None, None))
        else:
            rows.append((tag, scalar.step, scalar.value))

    for tag, step, value in rows:
        if value is None:
            print(f"MISSING {tag}")
        else:
            print(f"{tag}: step={step} value={value:.12f}")


if __name__ == "__main__":
    main()
