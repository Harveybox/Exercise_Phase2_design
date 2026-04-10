#!/usr/bin/env python3
import os
import csv
import glob
import argparse
from collections import OrderedDict


def read_csv_as_map(csv_path, key="description"):
    with open(csv_path, newline="") as fin:
        reader = csv.DictReader(fin)
        rows = OrderedDict()
        for row in reader:
            k = row.get(key, "")
            if not k:
                continue
            rows[k] = row
        return rows, list(reader.fieldnames or [])


def merge_rows(left, right):
    out = dict(left)
    for k, v in right.items():
        if k == "description":
            continue
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_root", required=True,
                    help="directory containing shard result folders, e.g. .../afcyc_out/shard_results")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    af_csvs = sorted(glob.glob(os.path.join(args.shard_root, "runlist_*", "results.csv")))
    if not af_csvs:
        raise SystemExit(f"No shard results.csv found under {args.shard_root}")

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged_rows = []
    all_fields = ["description"]

    for af_csv in af_csvs:
        shard_dir = os.path.dirname(af_csv)
        shard_name = os.path.basename(shard_dir)
        rmsd_csv = os.path.join(shard_dir, "rmsd_results.csv")

        af_map, af_fields = read_csv_as_map(af_csv)
        rmsd_map, rmsd_fields = (read_csv_as_map(rmsd_csv) if os.path.exists(rmsd_csv) else ({}, []))

        for field in af_fields + rmsd_fields + ["shard"]:
            if field and field not in all_fields:
                all_fields.append(field)

        seen = set(af_map) | set(rmsd_map)
        for desc in seen:
            left = af_map.get(desc, {"description": desc})
            right = rmsd_map.get(desc, {})
            row = merge_rows(left, right)
            row["shard"] = shard_name
            merged_rows.append(row)

    with open(args.out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=all_fields)
        writer.writeheader()
        for row in merged_rows:
            out_row = {k: row.get(k, "") for k in all_fields}
            writer.writerow(out_row)

    print(f"Saved merged csv: {args.out_csv}")
    print(f"Rows: {len(merged_rows)}")
    print(f"Shards merged: {len(af_csvs)}")


if __name__ == "__main__":
    main()
