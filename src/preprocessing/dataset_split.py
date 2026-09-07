"""Lesion-wise (patient-wise) leakage-safe train/val/test split for HAM10000.

HAM10000 contains multiple images per `lesion_id`. Splitting at the image level lets the
same lesion appear in both train and test, inflating reported accuracy. This module groups
by `lesion_id` first, then splits groups so every image of a given lesion stays in a single
split.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def lesion_wise_split(metadata: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
                       seed: int = 42) -> dict:
    lesion_labels = metadata.groupby("lesion_id")["dx"].first()
    lesion_ids = lesion_labels.index.to_numpy()
    strat = lesion_labels.to_numpy()

    train_ids, temp_ids, train_strat, temp_strat = train_test_split(
        lesion_ids, strat, test_size=(val_size + test_size), random_state=seed, stratify=strat
    )
    relative_test_size = test_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=relative_test_size, random_state=seed, stratify=temp_strat
    )

    return {
        "train": metadata[metadata["lesion_id"].isin(train_ids)],
        "val": metadata[metadata["lesion_id"].isin(val_ids)],
        "test": metadata[metadata["lesion_id"].isin(test_ids)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--out", required=True, help="Output directory for split CSVs")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    splits = lesion_wise_split(metadata, args.val_size, args.test_size, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} images, {df['lesion_id'].nunique()} lesions")


if __name__ == "__main__":
    main()
