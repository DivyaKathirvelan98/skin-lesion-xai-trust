"""Reorganize the raw Kaggle downloads into the canonical data/raw/ layout expected by
src/preprocessing/dataset.py, regardless of the exact nested folder structure Kaggle unzips to.

Target layout:
    <raw-dir>/images/<image_id>.jpg
    <raw-dir>/HAM10000_metadata.csv
    <raw-dir>/segmentations/<image_id>_segmentation.png
"""
import argparse
import shutil
from pathlib import Path


def consolidate_images(raw_dir: Path) -> int:
    images_dir = raw_dir / "images"
    images_dir.mkdir(exist_ok=True)
    count = 0
    for jpg in raw_dir.rglob("*.jpg"):
        if images_dir in jpg.parents:
            continue
        dest = images_dir / jpg.name
        if not dest.exists():
            shutil.move(str(jpg), str(dest))
            count += 1
    return count


def locate_metadata(raw_dir: Path) -> Path | None:
    candidates = [p for p in raw_dir.rglob("*metadata*.csv") if "segmentation" not in p.name.lower()]
    if not candidates:
        return None
    dest = raw_dir / "HAM10000_metadata.csv"
    if candidates[0] != dest:
        shutil.copy(str(candidates[0]), str(dest))
    return dest


def consolidate_segmentations(raw_dir: Path) -> int:
    seg_dir = raw_dir / "segmentations"
    seg_dir.mkdir(exist_ok=True)
    count = 0
    for mask in raw_dir.rglob("*segmentation*.png"):
        if seg_dir in mask.parents:
            continue
        dest = seg_dir / mask.name
        if not dest.exists():
            shutil.move(str(mask), str(dest))
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    n_images = consolidate_images(raw_dir)
    metadata = locate_metadata(raw_dir)
    n_masks = consolidate_segmentations(raw_dir)

    print(f"Consolidated {n_images} images into {raw_dir / 'images'}")
    print(f"Metadata: {metadata if metadata else 'NOT FOUND -- check the Kaggle download'}")
    print(f"Consolidated {n_masks} segmentation masks into {raw_dir / 'segmentations'}")


if __name__ == "__main__":
    main()
