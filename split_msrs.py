"""Copy a reproducible train/eval split from the MSRS training image pairs."""

import argparse
from pathlib import Path
import random
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset/MSRS-main"))
    parser.add_argument("--output", type=Path, default=Path("dataset/MSRS_split"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=122345)
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be between 0 and 1.")

    train_root = args.source / "train"
    folder_pairs = [("Visible", "Infrared"), ("vi", "ir")]
    available = [
        (train_root / visible, train_root / infrared)
        for visible, infrared in folder_pairs
        if (train_root / visible).is_dir() and (train_root / infrared).is_dir()
    ]
    if len(available) != 1:
        parser.error("Expected exactly one paired folder layout: train/vi + train/ir, or train/Visible + train/Infrared.")
    visible_root, infrared_root = available[0]

    visible_names = {p.name for p in visible_root.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
    infrared_names = {p.name for p in infrared_root.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
    if visible_names != infrared_names:
        parser.error(
            "Unpaired PNG files. Missing infrared: {}; missing visible: {}.".format(
                sorted(visible_names - infrared_names)[:5],
                sorted(infrared_names - visible_names)[:5],
            )
        )
    if len(visible_names) < 2:
        parser.error("At least two paired PNG images are required.")

    names = sorted(visible_names)
    eval_count = max(1, min(len(names) - 1, round(len(names) * args.val_ratio)))
    eval_names = set(random.Random(args.seed).sample(names, eval_count))
    train_names = visible_names - eval_names
    partitions = {"train": train_names, "eval": eval_names}

    # Reject existing output so a rerun cannot mix different partitions.
    args.output.mkdir(parents=True, exist_ok=False)
    for phase, selected_names in partitions.items():
        for folder, source_root in [("Visible", visible_root), ("Infrared", infrared_root)]:
            destination = args.output / phase / folder
            destination.mkdir(parents=True)
            for name in sorted(selected_names):
                shutil.copy2(source_root / name, destination / name)
        (args.output / (phase + "_files.txt")).write_text(
            "\n".join(sorted(selected_names)) + "\n", encoding="utf-8"
        )

    print("Source pairs: {}".format(len(names)))
    print("Training pairs: {}; evaluation pairs: {}".format(len(train_names), len(eval_names)))
    print("Output: {}".format(args.output.resolve()))
    print("Seed: {}".format(args.seed))
    print("Use --dataset_path {} --max_val_samples 0".format(args.output.as_posix()))


if __name__ == "__main__":
    main()
