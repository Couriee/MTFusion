"""Evaluate saved infrared-visible fusion images with the Table 1 metrics."""

import argparse
import csv
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from metric.fusion_metrics import AG_function, EN_function, SCD_function, SD_function, SF_function


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
METRICS = ("EN", "SD", "SF", "AG", "SCD")


def index_images(directory):
    images = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem.casefold()
        if stem in images:
            raise ValueError("Duplicate image names in {}: {}, {}".format(
                directory, images[stem].name, path.name
            ))
        images[stem] = path
    if not images:
        raise ValueError("No supported images in {}".format(directory))
    return images


def read_grayscale(path):
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    image = cv2.imdecode(encoded, flags) if encoded.size else None
    if image is None:
        raise ValueError("Could not decode {}".format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)


def evaluate(visible_dir, infrared_dir, fused_dir):
    directories = (visible_dir, infrared_dir, fused_dir)
    indexes = [index_images(directory) for directory in directories]
    stems = set().union(*indexes)
    missing = []
    for label, index in zip(("visible", "infrared", "fused"), indexes):
        absent = sorted(stems - index.keys())
        if absent:
            missing.append("{} missing {} ({})".format(label, len(absent), ", ".join(absent[:5])))
    if missing:
        raise ValueError("Image names do not match: {}".format("; ".join(missing)))

    rows = []
    for stem in sorted(stems):
        paths = [index[stem] for index in indexes]
        visible, infrared, fused = (read_grayscale(path) for path in paths)
        if visible.shape != infrared.shape or visible.shape != fused.shape:
            raise ValueError("Size mismatch for {}: visible {}, infrared {}, fused {}".format(
                paths[2].name, visible.shape, infrared.shape, fused.shape
            ))
        if min(fused.shape) < 2:
            raise ValueError("Image must be at least 2 by 2: {}".format(paths[2]))
        # A zero-variance image or residual makes a correlation term undefined.
        with np.errstate(divide="ignore", invalid="ignore"):
            scd = SCD_function(visible, infrared, fused)
        rows.append((paths[2].name, EN_function(fused), SD_function(fused),
                     SF_function(fused), AG_function(fused), scd))
    return rows


def write_results(output, rows):
    values = np.asarray([row[1:] for row in rows])
    valid_scd = values[np.isfinite(values[:, -1]), -1]
    means = list(values[:, :-1].mean(axis=0))
    means.append(valid_scd.mean() if valid_scd.size else math.nan)

    output.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                              dir=str(output.parent), delete=False)
    temporary_path = Path(file.name)
    try:
        with file:
            writer = csv.writer(file)
            writer.writerow(("image",) + METRICS)
            for name, *scores in rows:
                writer.writerow([name] + ["{:.10g}".format(score) for score in scores])
            writer.writerow(["MEAN"] + ["{:.10g}".format(score) for score in means])
        temporary_path.replace(output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print("Evaluated {} image triplets. SCD defined for {}/{}.".format(
        len(rows), valid_scd.size, len(rows)
    ))
    print("  ".join("{}={:.4f}".format(name, mean) for name, mean in zip(METRICS, means)))
    print("Saved per-image scores and means to {}".format(output))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Images are paired by filename stem and decoded as 8-bit grayscale. "
               "Undefined SCD scores are written as nan and excluded from its mean.",
    )
    parser.add_argument("--visible-dir", type=Path, required=True, help="visible source images")
    parser.add_argument("--infrared-dir", type=Path, required=True, help="infrared source images")
    parser.add_argument("--fused-dir", type=Path, required=True, help="saved fusion images")
    parser.add_argument("--output", type=Path, required=True, help="output CSV with a final MEAN row")
    args = parser.parse_args()

    try:
        rows = evaluate(args.visible_dir, args.infrared_dir, args.fused_dir)
        write_results(args.output, rows)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, "Evaluation failed: {}\n".format(error))


if __name__ == "__main__":
    main()
