"""Stratified subsampling of already lesion-wise-split CSVs, for a compute-limited (CPU-only)
reduced-scope run. Subsampling rows after the lesion-wise split cannot reintroduce leakage --
it only removes images, it never merges lesions across splits.
"""
import argparse
from pathlib import Path

import pandas as pd


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    frac = n / len(df)
    parts = [group.sample(max(1, round(len(group) * frac)), random_state=seed)
             for _, group in df.groupby("dx")]
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--train-n", type=int, default=800)
    parser.add_argument("--val-n", type=int, default=200)
    parser.add_argument("--test-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    targets = {"train": args.train_n, "val": args.val_n, "test": args.test_n}
    for split_name, n in targets.items():
        df = pd.read_csv(splits_dir / f"{split_name}.csv")
        sub = stratified_subsample(df, n, args.seed)
        sub.to_csv(splits_dir / f"{split_name}.csv", index=False)
        print(f"{split_name}: {len(sub)} images (from {len(df)})")
        print(sub["dx"].value_counts().to_dict())


if __name__ == "__main__":
    main()
