#!/usr/bin/env python3
import os
import csv
import glob
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_root", required=True,
                    help="directory containing shard result folders, e.g. .../afcyc_out/shard_results")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.shard_root, "runlist_*", "results.csv")))
    if not csvs:
        raise SystemExit(f"No shard results.csv found under {args.shard_root}")

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    seen = set()
    fieldnames = None
    rows_written = 0

    with open(args.out_csv, "w", newline="") as fout:
        writer = None

        for csv_path in csvs:
            shard_name = os.path.basename(os.path.dirname(csv_path))

            with open(csv_path, newline="") as fin:
                reader = csv.DictReader(fin)

                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                    if "shard" not in fieldnames:
                        fieldnames.append("shard")
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()

                for row in reader:
                    row["shard"] = shard_name

                    desc = row.get("description", "")
                    if desc and desc in seen:
                        continue
                    if desc:
                        seen.add(desc)

                    # 保证所有列都存在
                    out_row = {k: row.get(k, "") for k in fieldnames}
                    writer.writerow(out_row)
                    rows_written += 1

    print(f"Saved merged csv: {args.out_csv}")
    print(f"Rows: {rows_written}")
    print(f"Shards merged: {len(csvs)}")

if __name__ == "__main__":
    main()