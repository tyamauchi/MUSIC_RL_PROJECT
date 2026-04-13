"""
Simple utility to parse TensorBoard event files or log directories and print scalar summaries.
Usage:
    python utils/parse_tensorboard.py /path/to/events.out.tfevents...  # single event file
    python utils/parse_tensorboard.py /path/to/log_dir                  # directory with events files
"""
import os
import sys
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print("tensorboard is required to run this script. Install with: pip install tensorboard")
    raise


def summarize_log(path):
    # If path is a file, use that; if directory, pass dir to EventAccumulator
    if os.path.isfile(path):
        dirpath = os.path.dirname(path)
    else:
        dirpath = path

    ea = EventAccumulator(dirpath)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if not tags:
        print('No scalar tags found in the specified path.')
        return

    for tag in tags:
        events = ea.Scalars(tag)
        values = [e.value for e in events]
        steps = [e.step for e in events]
        if not values:
            continue
        import math
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        print(f"Tag: {tag}")
        print(f"  Count: {len(values)} | Steps: [{min(steps)}..{max(steps)}]")
        print(f"  Last: {values[-1]:.4f} | Mean: {mean:.4f} | Std: {std:.4f} | Min: {min(values):.4f} | Max: {max(values):.4f}")
        print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python utils/parse_tensorboard.py /path/to/event_file_or_log_dir')
        sys.exit(1)
    path = sys.argv[1]
    summarize_log(path)
