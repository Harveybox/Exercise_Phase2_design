#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_root", required=True,
                    help="directory containing shard result folders, e.g. .../pyrosetta_scores/shard_results")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.shard_root, "runlist_*", "pyrosetta_scores.csv")))
    if not csvs:
        raise SystemExit(f"No shard pyrosetta_scores.csv found under {args.shard_root}")

    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df["shard"] = os.path.basename(os.path.dirname(csv))
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    if "description" in merged.columns:
        merged = merged.drop_duplicates(subset=["description"], keep="first")

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(args.out_csv, index=False)
    print(f"Saved merged csv: {args.out_csv}")
    print(f"Rows: {len(merged)}")
    print(f"Shards merged: {len(csvs)}")

if __name__ == "__main__":
    main()