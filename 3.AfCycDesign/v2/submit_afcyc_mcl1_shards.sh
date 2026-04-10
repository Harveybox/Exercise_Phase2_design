#!/bin/bash
set -euo pipefail

# ====== AfCyc (GPU) ======
GPU_QUEUE="${GPU_QUEUE:-8v100-32-sc}"
GPU_NCPU="${GPU_NCPU:-1}"
GPU_PTILE="${GPU_PTILE:-1}"
GPU_REQ='num=1/host'

# ====== RMSD (CPU) ======
CPU_QUEUE="${CPU_QUEUE:-33}"
CPU_NCPU="${CPU_NCPU:-1}"
CPU_PTILE="${CPU_PTILE:-1}"

# ====== Common ======
N_SHARDS="${N_SHARDS:-15}"
MICROMAMBA="$HOME/bin/micromamba"
ENV_NAME="afcycdesign"
AFCYC_SCRIPT="$HOME/Exercise_Phase2_design/afcyc_predict_batch.py"
RMSD_SCRIPT="$HOME/Exercise_Phase2_design/rmsd_from_afcyc.py"
MERGE_SCRIPT="$HOME/Exercise_Phase2_design/merge_afcyc_csvs.py"

INPUT_DIR="/scratch/2026-03-28/bme-yaozm/mcl1_test/mpnn_relax4_out"
OUT_BASE="/scratch/2026-03-28/bme-yaozm/mcl1_test/afcyc_out"
AF_PARAMS_DIR="$HOME/dl_binder_design/af2_initial_guess/model_weights/params"

TARGET_CHAIN="A"
BINDER_CHAIN="B"

RUNLIST_DIR="$OUT_BASE/runlists"
SHARD_INPUT_ROOT="$OUT_BASE/shard_inputs"
SHARD_RESULT_ROOT="$OUT_BASE/shard_results"
LOG_DIR="$OUT_BASE/logs"
MERGED_DIR="$OUT_BASE/merged"

mkdir -p "$RUNLIST_DIR" "$SHARD_INPUT_ROOT" "$SHARD_RESULT_ROOT" "$LOG_DIR" "$MERGED_DIR"

ALL_TAGS="$RUNLIST_DIR/all_tags.txt"

# 1) collect all design pdb names
python3 - <<'PY' "$INPUT_DIR" "$ALL_TAGS"
import sys, os, glob
input_dir, out_file = sys.argv[1], sys.argv[2]
pdbs = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
if not pdbs:
    raise SystemExit(f"No pdb files found in {input_dir}")
with open(out_file, "w") as f:
    for p in pdbs:
        f.write(os.path.basename(p) + "\n")
print(f"Wrote {len(pdbs)} pdb names to {out_file}")
PY

# 2) split into shards
python3 - <<'PY' "$ALL_TAGS" "$RUNLIST_DIR" "$N_SHARDS"
import sys, os, math
all_tags_file, runlist_dir, n_shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(all_tags_file) as f:
    names = [x.strip() for x in f if x.strip()]
n = len(names)
chunk = math.ceil(n / n_shards)
for fn in os.listdir(runlist_dir):
    if fn.startswith("runlist_") and fn.endswith(".txt"):
        os.remove(os.path.join(runlist_dir, fn))
for i in range(n_shards):
    sub = names[i*chunk:(i+1)*chunk]
    if not sub:
        continue
    out = os.path.join(runlist_dir, f"runlist_{i:02d}.txt")
    with open(out, "w") as f:
        for x in sub:
            f.write(x + "\n")
    print(f"{out}: {len(sub)} files")
PY

# 3) create shard input dirs of symlinks
for RUNLIST in "$RUNLIST_DIR"/runlist_*.txt; do
    shard=$(basename "$RUNLIST" .txt)
    SHARD_INPUT_DIR="$SHARD_INPUT_ROOT/$shard"
    mkdir -p "$SHARD_INPUT_DIR"
    find "$SHARD_INPUT_DIR" -maxdepth 1 -type l -name "*.pdb" -delete
    while IFS= read -r pdb_name; do
        ln -sf "$INPUT_DIR/$pdb_name" "$SHARD_INPUT_DIR/$pdb_name"
    done < "$RUNLIST"
done

AFCYC_JOB_IDS=()
RMSD_JOB_IDS=()

# 4) submit each AfCyc shard and immediately chain a CPU RMSD shard job to it
for RUNLIST in "$RUNLIST_DIR"/runlist_*.txt; do
    shard=$(basename "$RUNLIST" .txt)
    SHARD_INPUT_DIR="$SHARD_INPUT_ROOT/$shard"
    SHARD_OUT_DIR="$SHARD_RESULT_ROOT/$shard"
    mkdir -p "$SHARD_OUT_DIR"

    af_submit=$(bsub <<EOF
#!/bin/bash
#BSUB -J afcyc_${shard}
#BSUB -q ${GPU_QUEUE}
#BSUB -n ${GPU_NCPU}
#BSUB -R "span[ptile=${GPU_PTILE}]"
#BSUB -gpu "${GPU_REQ}"
#BSUB -o ${LOG_DIR}/${shard}.afcyc.%J.out
#BSUB -e ${LOG_DIR}/${shard}.afcyc.%J.err
set -euo pipefail
date
${MICROMAMBA} run -n "${ENV_NAME}" python "${AFCYC_SCRIPT}" \
  --input_dir "${SHARD_INPUT_DIR}" \
  --out_dir "${SHARD_OUT_DIR}" \
  --data_dir "${AF_PARAMS_DIR}" \
  --target_chain "${TARGET_CHAIN}" \
  --binder_chain "${BINDER_CHAIN}" \
  --offset_type 2 \
  --num_recycles 3 \
  --num_models 1
date
EOF
)
    echo "$af_submit"
    af_job_id=$(echo "$af_submit" | sed -n 's/Job <\([0-9]\+\)>.*/\1/p')
    if [[ -n "$af_job_id" ]]; then
        AFCYC_JOB_IDS+=("$af_job_id")
    fi

    rmsd_submit=$(bsub -w "done(${af_job_id})" <<EOF
#!/bin/bash
#BSUB -J rmsd_${shard}
#BSUB -q ${CPU_QUEUE}
#BSUB -n ${CPU_NCPU}
#BSUB -R "span[ptile=${CPU_PTILE}]"
#BSUB -o ${LOG_DIR}/${shard}.rmsd.%J.out
#BSUB -e ${LOG_DIR}/${shard}.rmsd.%J.err
set -euo pipefail
date
${MICROMAMBA} run -n "${ENV_NAME}" python "${RMSD_SCRIPT}" \
  --design_dir "${SHARD_INPUT_DIR}" \
  --pred_dir "${SHARD_OUT_DIR}/pred_pdbs" \
  --out_csv "${SHARD_OUT_DIR}/rmsd_results.csv" \
  --target_chain "${TARGET_CHAIN}" \
  --binder_chain "${BINDER_CHAIN}"
date
EOF
)
    echo "$rmsd_submit"
    rmsd_job_id=$(echo "$rmsd_submit" | sed -n 's/Job <\([0-9]\+\)>.*/\1/p')
    if [[ -n "$rmsd_job_id" ]]; then
        RMSD_JOB_IDS+=("$rmsd_job_id")
    fi
done

# 5) merge after all RMSD shard jobs complete
if [[ ${#RMSD_JOB_IDS[@]} -gt 0 ]]; then
    DEP=$(printf "done(%s) && " "${RMSD_JOB_IDS[@]}")
    DEP=${DEP% && }
    MERGED_CSV="$MERGED_DIR/results_merged.csv"

    bsub -w "$DEP" <<EOF
#!/bin/bash
#BSUB -J afcyc_merge
#BSUB -q ${CPU_QUEUE}
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -o ${LOG_DIR}/merge.%J.out
#BSUB -e ${LOG_DIR}/merge.%J.err
set -euo pipefail
date
${MICROMAMBA} run -n "${ENV_NAME}" python "${MERGE_SCRIPT}" \
  --shard_root "${SHARD_RESULT_ROOT}" \
  --out_csv "${MERGED_CSV}"
date
EOF
fi

echo
echo "Submitted ${#AFCYC_JOB_IDS[@]} AfCyc shard jobs."
echo "Submitted ${#RMSD_JOB_IDS[@]} RMSD shard jobs."
echo "Shard inputs:  $SHARD_INPUT_ROOT"
echo "Shard outputs: $SHARD_RESULT_ROOT"
echo "Merged csv:    $MERGED_DIR/results_merged.csv"
