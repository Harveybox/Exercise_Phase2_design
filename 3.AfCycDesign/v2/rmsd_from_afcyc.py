#!/usr/bin/env python3
import csv
import math
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
from Bio.PDB import PDBParser, is_aa


PARSER = PDBParser(QUIET=True)


def load_model(pdb_file: str):
    structure = PARSER.get_structure("x", pdb_file)
    return next(structure.get_models())


def chain_lengths(pdb_file: str):
    model = load_model(pdb_file)
    out = OrderedDict()
    for chain in model:
        n = 0
        for res in chain:
            if is_aa(res, standard=True):
                n += 1
        if n > 0:
            out[chain.id] = n
    return out


def pick_chain_id(pdb_file: str, requested_chain: str, fallback: str):
    model = load_model(pdb_file)
    if requested_chain in model:
        return requested_chain
    lengths = chain_lengths(pdb_file)
    if not lengths:
        raise ValueError(f"No protein chains found in {pdb_file}")
    if fallback == "shortest":
        return min(lengths, key=lengths.get)
    if fallback == "longest":
        return max(lengths, key=lengths.get)
    raise ValueError(f"Unknown fallback: {fallback}")


def extract_ca_coords(pdb_file: str, chain_id: str, fallback: str):
    model = load_model(pdb_file)
    picked = pick_chain_id(pdb_file, chain_id, fallback)
    chain = model[picked]
    coords = []
    for res in chain:
        if is_aa(res, standard=True) and "CA" in res:
            coords.append(res["CA"].coord)
    return np.array(coords, dtype=float), picked


def calc_rmsd(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape or x.size == 0:
        return math.nan
    diff = x - y
    return float(np.sqrt((diff * diff).sum() / len(x)))


def kabsch_align(mobile: np.ndarray, target: np.ndarray):
    if mobile.shape != target.shape or mobile.size == 0:
        raise ValueError("Coordinate arrays must have same non-zero shape")
    mob_cent = mobile.mean(axis=0)
    tar_cent = target.mean(axis=0)
    mob0 = mobile - mob_cent
    tar0 = target - tar_cent
    h = mob0.T @ tar0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = tar_cent - mob_cent @ r
    return r, t


def apply_transform(coords: np.ndarray, r: np.ndarray, t: np.ndarray):
    return coords @ r + t


def safe_super_rmsd(mobile: np.ndarray, target: np.ndarray):
    if mobile.shape != target.shape or mobile.size == 0:
        return math.nan, None, None
    r, t = kabsch_align(mobile, target)
    aligned = apply_transform(mobile, r, t)
    return calc_rmsd(aligned, target), r, t


def build_row(design_pdb: str, pred_pdb: str, target_chain: str, binder_chain: str):
    design_target, design_target_chain = extract_ca_coords(design_pdb, target_chain, "longest")
    pred_target, pred_target_chain = extract_ca_coords(pred_pdb, target_chain, "longest")
    design_binder, design_binder_chain = extract_ca_coords(design_pdb, binder_chain, "shortest")
    pred_binder, pred_binder_chain = extract_ca_coords(pred_pdb, binder_chain, "shortest")

    binder_no_align = calc_rmsd(design_binder, pred_binder)
    binder_only_rmsd, _, _ = safe_super_rmsd(pred_binder, design_binder)
    target_self_rmsd, r_target, t_target = safe_super_rmsd(pred_target, design_target)

    if r_target is None:
        binder_target_align = math.nan
    else:
        pred_binder_on_design = apply_transform(pred_binder, r_target, t_target)
        binder_target_align = calc_rmsd(pred_binder_on_design, design_binder)

    design_complex = np.vstack([design_target, design_binder])
    pred_complex = np.vstack([pred_target, pred_binder])
    complex_global_rmsd, _, _ = safe_super_rmsd(pred_complex, design_complex)

    return {
        "description": Path(design_pdb).stem,
        "input_pdb": str(design_pdb),
        "pred_pdb": str(pred_pdb),
        "target_chain_input": target_chain,
        "binder_chain_input": binder_chain,
        "target_chain_pred": pred_target_chain,
        "binder_chain_pred": pred_binder_chain,
        "binder_ca_rmsd_no_align": binder_no_align,
        "binder_ca_rmsd_target_align": binder_target_align,
        "binder_ca_rmsd_binder_align": binder_only_rmsd,
        "target_ca_rmsd_target_align": target_self_rmsd,
        "complex_ca_rmsd_global_align": complex_global_rmsd,
        "status": "ok",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--target_chain", required=True)
    ap.add_argument("--binder_chain", required=True)
    args = ap.parse_args()

    design_dir = Path(args.design_dir)
    pred_dir = Path(args.pred_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    design_pdbs = sorted(design_dir.glob("*.pdb"))
    if not design_pdbs:
        raise SystemExit(f"No pdb files found in {design_dir}")

    rows = []
    for design_pdb in design_pdbs:
        desc = design_pdb.stem
        pred_pdb = pred_dir / f"{desc}_afcyc.pdb"
        if not pred_pdb.exists() or pred_pdb.stat().st_size == 0:
            rows.append({
                "description": desc,
                "input_pdb": str(design_pdb),
                "pred_pdb": str(pred_pdb),
                "target_chain_input": args.target_chain,
                "binder_chain_input": args.binder_chain,
                "target_chain_pred": "",
                "binder_chain_pred": "",
                "binder_ca_rmsd_no_align": "",
                "binder_ca_rmsd_target_align": "",
                "binder_ca_rmsd_binder_align": "",
                "target_ca_rmsd_target_align": "",
                "complex_ca_rmsd_global_align": "",
                "status": "error: missing pred_pdb",
            })
            print(f"[MISS] {desc}: {pred_pdb}")
            continue

        try:
            row = build_row(str(design_pdb), str(pred_pdb), args.target_chain, args.binder_chain)
            rows.append(row)
            print(
                f"[OK] {desc}  target_align={row['binder_ca_rmsd_target_align']:.3f}  "
                f"binder_align={row['binder_ca_rmsd_binder_align']:.3f}"
            )
        except Exception as e:
            rows.append({
                "description": desc,
                "input_pdb": str(design_pdb),
                "pred_pdb": str(pred_pdb),
                "target_chain_input": args.target_chain,
                "binder_chain_input": args.binder_chain,
                "target_chain_pred": "",
                "binder_chain_pred": "",
                "binder_ca_rmsd_no_align": "",
                "binder_ca_rmsd_target_align": "",
                "binder_ca_rmsd_binder_align": "",
                "target_ca_rmsd_target_align": "",
                "complex_ca_rmsd_global_align": "",
                "status": f"error: {e}",
            })
            print(f"[ERROR] {desc}: {e}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
