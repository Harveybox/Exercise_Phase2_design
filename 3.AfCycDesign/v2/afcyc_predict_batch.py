#!/usr/bin/env python3
import csv
import math
import argparse
import traceback
from pathlib import Path
from collections import OrderedDict

import numpy as np
from Bio.PDB import PDBParser, is_aa
from Bio.Data.IUPACData import protein_letters_3to1

from colabdesign import mk_afdesign_model, clear_mem
import jax
import jax.numpy as jnp
from colabdesign.af.alphafold.common import residue_constants


# ----------------------------
# monkey patches from af_cyc_design.ipynb
# ----------------------------
def add_cyclic_offset(self, offset_type=2):
    """Add cyclic offset to connect N and C term."""

    def cyclic_offset(length):
        i = np.arange(length)
        ij = np.stack([i, i + length], -1)
        offset = i[:, None] - i[None, :]
        c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
        if offset_type == 1:
            pass
        elif offset_type >= 2:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
            if offset_type == 3:
                idx = np.abs(c_offset) > 2
                c_offset[idx] = (32 * c_offset[idx]) / np.abs(c_offset[idx])
        return c_offset * np.sign(offset)

    idx = self._inputs["residue_index"]
    offset = np.array(idx[:, None] - idx[None, :])

    if self.protocol == "binder":
        c_offset = cyclic_offset(self._binder_len)
        offset[self._target_len:, self._target_len:] = c_offset

    if self.protocol in ["fixbb", "partial", "hallucination"]:
        ln = 0
        for length in self._lengths:
            offset[ln:ln + length, ln:ln + length] = cyclic_offset(length)
            ln += length

    self._inputs["offset"] = offset


def add_rg_loss(self, weight=0.1):
    """Optional, mainly useful for hallucination; not needed for this pipeline."""

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365
        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight


# ----------------------------
# PDB helpers
# ----------------------------
def aa3_to_1(resname: str) -> str:
    resname = resname.upper()
    return protein_letters_3to1.get(resname.capitalize(), "X")


def load_model(pdb_file: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_file)
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


def detect_binder_chain(pdb_file: str) -> str:
    lengths = chain_lengths(pdb_file)
    if not lengths:
        raise ValueError(f"No protein chains found in {pdb_file}")
    return min(lengths, key=lengths.get)


def detect_target_chain(pdb_file: str, binder_chain: str) -> str:
    lengths = chain_lengths(pdb_file)
    other = {k: v for k, v in lengths.items() if k != binder_chain}
    if not other:
        raise ValueError(f"No target chain found in {pdb_file}")
    return max(other, key=other.get)


def get_chain_seq(pdb_file: str, chain_id: str) -> str:
    model = load_model(pdb_file)
    chain = model[chain_id]
    seq = []
    for res in chain:
        if is_aa(res, standard=True):
            seq.append(aa3_to_1(res.get_resname()))
    return "".join(seq)


def maybe_scalar(x):
    if x is None:
        return math.nan
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    x = np.asarray(x)
    if x.size == 0:
        return math.nan
    return float(x.mean())


def extract_metric_from_aux(aux: dict, names):
    for key in names:
        if key in aux:
            return maybe_scalar(aux[key])

    if "log" in aux and isinstance(aux["log"], dict):
        for key in names:
            if key in aux["log"]:
                return maybe_scalar(aux["log"][key])

    return math.nan


def extract_binder_plddt(aux: dict, target_len: int, binder_len: int):
    direct = extract_metric_from_aux(aux, ["plddt_binder", "binder_plddt"])
    if not math.isnan(direct):
        return direct

    plddt = None
    if "plddt" in aux:
        plddt = np.asarray(aux["plddt"])
    elif "log" in aux and isinstance(aux["log"], dict) and "plddt" in aux["log"]:
        plddt = np.asarray(aux["log"]["plddt"])

    if plddt is None or plddt.size == 0:
        return math.nan

    plddt = plddt.reshape(-1)
    if plddt.size < target_len + binder_len:
        return maybe_scalar(plddt)

    return float(plddt[target_len:target_len + binder_len].mean())


# ----------------------------
# ColabDesign helpers
# ----------------------------
def save_pred_pdb(model, out_pdb):
    if hasattr(model, "save_pdb"):
        model.save_pdb(str(out_pdb))
        return
    if hasattr(model, "save_current_pdb"):
        model.save_current_pdb(str(out_pdb))
        return
    raise RuntimeError("Neither save_pdb() nor save_current_pdb() found on model")


def set_fixed_sequence_and_predict(model, binder_seq: str):
    last_err = None

    if hasattr(model, "predict"):
        try:
            model.predict(seq=binder_seq)
            return
        except Exception as e:
            last_err = e

    if hasattr(model, "set_seq") and hasattr(model, "predict"):
        try:
            model.set_seq(seq=binder_seq)
            model.predict()
            return
        except Exception as e:
            last_err = e

    if hasattr(model, "set_seq") and hasattr(model, "predict"):
        try:
            model.set_seq(mode="wildtype")
            model.predict()
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not find a working fixed-sequence prediction call in this local ColabDesign build. "
        f"Last error: {last_err}"
    )


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", required=True, help="directory containing AlphaFold params")
    ap.add_argument("--target_chain", default=None)
    ap.add_argument("--binder_chain", default=None)
    ap.add_argument("--offset_type", type=int, default=2)
    ap.add_argument("--num_recycles", type=int, default=3)
    ap.add_argument("--num_models", type=int, default=1)
    ap.add_argument("--use_multimer", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "pred_pdbs"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pdbs = sorted(input_dir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No pdb files found in {input_dir}")

    for pdb in pdbs:
        desc = pdb.stem
        try:
            binder_chain = args.binder_chain or detect_binder_chain(str(pdb))
            target_chain = args.target_chain or detect_target_chain(str(pdb), binder_chain)
            binder_seq = get_chain_seq(str(pdb), binder_chain)
            target_len = len(get_chain_seq(str(pdb), target_chain))
            binder_len = len(binder_seq)

            clear_mem()
            model = mk_afdesign_model(
                protocol="binder",
                data_dir=args.data_dir,
                num_recycles=args.num_recycles,
                num_models=args.num_models,
                use_multimer=args.use_multimer,
            )

            model.prep_inputs(
                pdb_filename=str(pdb),
                chain=target_chain,
                binder_chain=binder_chain,
            )
            add_cyclic_offset(model, offset_type=args.offset_type)
            set_fixed_sequence_and_predict(model, binder_seq)

            pred_pdb = pred_dir / f"{desc}_afcyc.pdb"
            save_pred_pdb(model, pred_pdb)

            i_pae = extract_metric_from_aux(model.aux, ["i_pae", "pae_interaction"])
            plddt_binder = extract_binder_plddt(model.aux, target_len, binder_len)

            rows.append({
                "description": desc,
                "input_pdb": str(pdb),
                "pred_pdb": str(pred_pdb),
                "target_chain": target_chain,
                "binder_chain_input": binder_chain,
                "target_len": target_len,
                "binder_len": binder_len,
                "binder_seq": binder_seq,
                "i_pae": i_pae,
                "plddt_binder": plddt_binder,
                "status": "ok",
            })
            print(f"[OK] {desc}  i_pae={i_pae:.4f}  plddt_binder={plddt_binder:.4f}")

        except Exception as e:
            rows.append({
                "description": desc,
                "input_pdb": str(pdb),
                "pred_pdb": "",
                "target_chain": args.target_chain or "",
                "binder_chain_input": args.binder_chain or "",
                "target_len": "",
                "binder_len": "",
                "binder_seq": "",
                "i_pae": "",
                "plddt_binder": "",
                "status": f"error: {e}",
            })
            print(f"[ERROR] {desc}: {e}")
            traceback.print_exc()

    out_csv = out_dir / "results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
