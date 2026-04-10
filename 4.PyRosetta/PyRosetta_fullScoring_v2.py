#!/usr/bin/env python3
import os
import csv
import glob
import math
import argparse
import faulthandler

faulthandler.enable()

import pyrosetta
from pyrosetta import pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import SapScoreMetric, PerResidueSapScoreMetric
from pyrosetta.rosetta.protocols.simple_filters import ContactMolecularSurfaceFilter

from Bio.PDB import PDBParser, is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.SeqUtils.ProtParam import ProteinAnalysis


SWI_WEIGHTS = {
    "A": 0.8356471476582918,
    "C": 0.5208088354857734,
    "D": 0.9079044671339564,
    "E": 0.9876987431418378,
    "F": 0.5849790194237692,
    "G": 0.7997168496420723,
    "H": 0.8947913996466419,
    "I": 0.6784124413866582,
    "K": 0.9267104557513497,
    "L": 0.6554221515081433,
    "M": 0.6296623675420369,
    "N": 0.8597433107431216,
    "P": 0.8235328714705341,
    "Q": 0.789434648348208,
    "R": 0.7712466317693457,
    "S": 0.7440908318492778,
    "T": 0.8096922697856334,
    "V": 0.7357837119163659,
    "W": 0.6374678690957594,
    "Y": 0.6112801822947587,
}
BULKY = set("FWYILVT")
TURN_PROMOTERS = set("GP")
CHARGED = set("DEKRH")
AROMATIC = set("FWY")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def aa3_to_1(resname: str) -> str:
    resname = resname.upper()
    return protein_letters_3to1.get(resname.capitalize(), "X")


def get_chain_seq_from_pdb(pdb_file: str, chain_id: str) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_file)
    model = next(structure.get_models())
    chain = model[chain_id]
    seq = []
    for res in chain:
        if is_aa(res, standard=True):
            seq.append(aa3_to_1(res.get_resname()))
    return "".join(seq)


def calc_swi(seq: str) -> float:
    vals = [SWI_WEIGHTS[aa] for aa in seq if aa in SWI_WEIGHTS]
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))


def calc_basic_seq_metrics(seq: str):
    if not seq:
        return {
            "binder_len": math.nan,
            "swi_score": math.nan,
            "gravy": math.nan,
            "aromaticity": math.nan,
            "isoelectric_point": math.nan,
            "instability_index": math.nan,
            "charge_pH7": math.nan,
            "frac_charged": math.nan,
            "frac_aromatic": math.nan,
            "n_term_bulky": math.nan,
            "c_term_bulky": math.nan,
            "junction_turn_count": math.nan,
            "cyclization_seq_feasibility": math.nan,
        }

    ana = ProteinAnalysis(seq)
    n_term = seq[0]
    c_term = seq[-1]
    junction_window = seq[:2] + seq[-2:]
    junction_turn_count = sum(1 for aa in junction_window if aa in TURN_PROMOTERS)

    score = 0.0
    n = len(seq)
    if 7 <= n <= 18:
        score += 2.0
    elif 5 <= n <= 24:
        score += 1.0
    else:
        score -= 1.0

    if n_term not in BULKY:
        score += 1.0
    else:
        score -= 0.5

    if c_term not in BULKY:
        score += 1.0
    else:
        score -= 0.5

    score += 0.5 * junction_turn_count
    if abs(ana.charge_at_pH(7.0)) <= 2.0:
        score += 0.5

    return {
        "binder_len": len(seq),
        "swi_score": calc_swi(seq),
        "gravy": safe_float(ana.gravy()),
        "aromaticity": safe_float(ana.aromaticity()),
        "isoelectric_point": safe_float(ana.isoelectric_point()),
        "instability_index": safe_float(ana.instability_index()),
        "charge_pH7": safe_float(ana.charge_at_pH(7.0)),
        "frac_charged": sum(1 for aa in seq if aa in CHARGED) / len(seq),
        "frac_aromatic": sum(1 for aa in seq if aa in AROMATIC) / len(seq),
        "n_term_bulky": float(n_term in BULKY),
        "c_term_bulky": float(c_term in BULKY),
        "junction_turn_count": float(junction_turn_count),
        "cyclization_seq_feasibility": float(score),
    }


def calc_interface_metrics(pose, scorefxn, interface_str, pack_input=True, pack_separated=True, packstat=True):
    iam = InterfaceAnalyzerMover(
        interface_str,
        False,
        scorefxn,
        packstat,
        pack_input,
        pack_separated,
        False,
        False,
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

    init_flags = "-mute all -ignore_unrecognized_res 1"
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
        if os.path.getsize(pdb) == 0:
            print(f"Skipping empty file: {pdb}")
            continue

        try:
            pose = pose_from_pdb(pdb)
            binder_seq = get_chain_seq_from_pdb(pdb, args.binder_chain)

            row = {
                "description": os.path.basename(pdb).replace(".pdb", ""),
                "pdb_path": pdb,
                "binder_seq": binder_seq,
            }
            row.update(calc_basic_seq_metrics(binder_seq))
            row.update(
                calc_interface_metrics(
                    pose=pose,
                    scorefxn=scorefxn,
                    interface_str=interface_str,
                    pack_input=args.pack_input,
                    pack_separated=args.pack_separated,
                    packstat=args.packstat,
                )
            )
            row["sap_binder"] = calc_sap_binder(pose, binder_sel)
            row["cms"] = calc_cms(pose, target_sel, binder_sel)
            rows.append(row)

            print(
                f"{row['description']} | i_dG={row['interface_dG']:.3f} | "
                f"dSASA={row['interface_delta_sasa']:.3f} | SAP={row['sap_binder']:.3f} | "
                f"CMS={row['cms']:.3f} | SWI={row['swi_score']:.3f} | cyc={row['cyclization_seq_feasibility']:.2f}"
            )
        except Exception as e:
            print(f"[ERROR] {pdb}: {e}")

    if not rows:
        raise SystemExit("No valid rows to write")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
