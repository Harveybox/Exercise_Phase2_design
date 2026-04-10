import os
import csv
import glob
import math
import argparse
import faulthandler # 新增
faulthandler.enable() # 新增：捕捉 SIGSEGV 的具体位置

import pyrosetta
from pyrosetta import pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import SapScoreMetric, PerResidueSapScoreMetric
from pyrosetta.rosetta.protocols.simple_filters import ContactMolecularSurfaceFilter


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def calc_interface_metrics(pose, scorefxn, interface_str, pack_input=True, pack_separated=True, packstat=True):
    iam = InterfaceAnalyzerMover(
        interface_str,
        False,
        scorefxn,
        packstat,
        pack_input,
        pack_separated,
        False,
        False
    )
    iam.apply(pose)

    return {
        "total_score": safe_float(scorefxn(pose)),
        "interface_dG": safe_float(iam.get_interface_dG()),
        "separated_interface_energy": safe_float(iam.get_separated_interface_energy()),
        "interface_delta_sasa": safe_float(iam.get_interface_delta_sasa()),
        "interface_packstat": safe_float(iam.get_interface_packstat()),
        "interface_delta_hbond_unsat": safe_float(iam.get_interface_delta_hbond_unsat()),
        "num_interface_residues": safe_float(iam.get_num_interface_residues()),
        "gly_interface_energy": safe_float(iam.get_gly_interface_energy()),
    }


def calc_sap_binder(pose, binder_sel):
    try:
        metric = SapScoreMetric()
        if hasattr(metric, "set_selector"):
            metric.set_selector(binder_sel)
        elif hasattr(metric, "selector"):
            metric.selector(binder_sel)
        return safe_float(metric.calculate(pose))
    except Exception:
        pass

    try:
        per_metric = PerResidueSapScoreMetric(binder_sel)
        values = per_metric.calculate(pose)
        total = 0.0
        if hasattr(values, "items"):
            for _, v in values.items():
                total += float(v)
            return total
        for k in values:
            total += float(values[k])
        return total
    except Exception:
        return math.nan


def calc_cms(pose, target_sel, binder_sel):
    try:
        cms = ContactMolecularSurfaceFilter()
        cms.selector1(target_sel)
        cms.selector2(binder_sel)
        if hasattr(cms, "compute"):
            return safe_float(cms.compute(pose))
        if hasattr(cms, "report_sm"):
            return safe_float(cms.report_sm(pose))
    except Exception:
        pass
    return math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--target_chain", required=True)
    ap.add_argument("--binder_chain", required=True)
    ap.add_argument("--pack_input", action="store_true")
    ap.add_argument("--pack_separated", action="store_true")
    ap.add_argument("--packstat", action="store_true")
    ap.add_argument("--extra_init", default="")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    init_flags = "-mute all"
    if args.extra_init.strip():
        init_flags += " " + args.extra_init.strip()
    pyrosetta.init(init_flags)

    scorefxn = get_fa_scorefxn()
    target_sel = ChainSelector(args.target_chain)
    binder_sel = ChainSelector(args.binder_chain)
    interface_str = f"{args.target_chain}_{args.binder_chain}"

    pdbs = sorted(glob.glob(os.path.join(args.input_dir, "*.pdb")))
    if not pdbs:
        raise SystemExit(f"No pdb files found in {args.input_dir}")

    rows = []
    for pdb in pdbs:
        pose = pose_from_pdb(pdb)

        row = {"description": os.path.basename(pdb).replace(".pdb", "")}
        row.update(calc_interface_metrics(
            pose=pose,
            scorefxn=scorefxn,
            interface_str=interface_str,
            pack_input=args.pack_input,
            pack_separated=args.pack_separated,
            packstat=args.packstat,
        ))
        row["sap_binder"] = calc_sap_binder(pose, binder_sel)
        row["cms"] = calc_cms(pose, target_sel, binder_sel)

        rows.append(row)

        print(
            f"{row['description']} | "
            f"i_dG={row['interface_dG']:.3f} | "
            f"dSASA={row['interface_delta_sasa']:.3f} | "
            f"SAP={row['sap_binder']:.3f} | "
            f"CMS={row['cms']:.3f}"
        )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.out_csv}")

    # 新增 -ignore_unrecognized_res 1 
    # RFpeptide 生成的是大环，首尾相连处如果不加此 flag，容易导致载入时崩溃
    init_flags = "-mute all -ignore_unrecognized_res 1"
    if args.extra_init.strip():
        init_flags += " " + args.extra_init.strip()
    pyrosetta.init(init_flags)

    # ... 中间代码不变 ...
    
    rows = []
    for pdb in pdbs:
        # 新增：检查文件是否为空（避免因 MPNN 失败产生的 0 字节文件导致崩溃）
        if os.path.getsize(pdb) == 0:
            print(f"Skipping empty file: {pdb}")
            continue
            
        pose = pose_from_pdb(pdb)

if __name__ == "__main__":
    main()