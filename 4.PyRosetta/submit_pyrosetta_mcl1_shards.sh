#!/bin/bash
set -euo pipefail

QUEUE="${QUEUE:-33}"
N_SHARDS="${N_SHARDS:-50}"

MICROMAMBA="$HOME/bin/micromamba"
ENV_NAME="PyRosettaScore"
PYTHON_BIN="$($MICROMAMBA run -n "$ENV_NAME" which python)"
SCRIPT="$HOME/Exercise_Phase2_design/PyRosetta_fullScoring.py"

INPUT_DIR="/scratch/2026-03-28/bme-yaozm/mcl1_test/mpnn_relax4_out"
OUT_BASE="/scratch/2026-03-28/bme-yaozm/mcl1_test/pyrosetta_scores"

TARGET_CHAIN="A"
BINDER_CHAIN="B"

NCPU=2
PTILE=2

RUNLIST_DIR="$OUT_BASE/runlists"
SHARD_INPUT_ROOT="$OUT_BASE/shard_inputs"
SHARD_RESULT_ROOT="$OUT_BASE/shard_results"
LOG_DIR="$OUT_BASE/logs"
MERGED_DIR="$OUT_BASE/merged"

mkdir -p "$RUNLIST_DIR" "$SHARD_INPUT_ROOT" "$SHARD_RESULT_ROOT" "$LOG_DIR" "$MERGED_DIR"

ALL_TAGS="$RUNLIST_DIR/all_pdbs.txt"

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

for RUNLIST in "$RUNLIST_DIR"/runlist_*.txt; do
    shard=$(basename "$RUNLIST" .txt)
    SHARD_INPUT_DIR="$SHARD_INPUT_ROOT/$shard"
    mkdir -p "$SHARD_INPUT_DIR"
    find "$SHARD_INPUT_DIR" -maxdepth 1 -type l -name "*.pdb" -delete

    while IFS= read -r pdb_name; do
        ln -sf "$INPUT_DIR/$pdb_name" "$SHARD_INPUT_DIR/$pdb_name"
    done < "$RUNLIST"
done

JOB_IDS=()

for RUNLIST in "$RUNLIST_DIR"/runlist_*.txt; do
    shard=$(basename "$RUNLIST" .txt)
    SHARD_INPUT_DIR="$SHARD_INPUT_ROOT/$shard"
    SHARD_OUT_DIR="$SHARD_RESULT_ROOT/$shard"
    SHARD_OUT_CSV="$SHARD_OUT_DIR/pyrosetta_scores.csv"

    mkdir -p "$SHARD_OUT_DIR"

    job_submit_output=$(bsub <<EOF
#!/bin/bash
#BSUB -J pyro_${shard}
#BSUB -q ${QUEUE}
#BSUB -n ${NCPU}
#BSUB -R "span[ptile=${PTILE}]"
#BSUB -o ${LOG_DIR}/${shard}.%J.out
#BSUB -e ${LOG_DIR}/${shard}.%J.err

set -euo pipefail
date

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

MICROMAMBA="$HOME/bin/micromamba"
ENV_NAME="PyRosettaScore"
SCRIPT="$HOME/Exercise_Phase2_design/PyRosetta_fullScoring.py"
PYTHON_BIN="${PYTHON_BIN}"
SHARD_INPUT_DIR="${SHARD_INPUT_DIR}"
SHARD_OUT_CSV="${SHARD_OUT_CSV}"
TARGET_CHAIN="A"
BINDER_CHAIN="B"

\$PYTHON_BIN "\$SCRIPT" \
  --input_dir "\$SHARD_INPUT_DIR" \
  --out_csv "\$SHARD_OUT_CSV" \
  --target_chain "\$TARGET_CHAIN" \
  --binder_chain "\$BINDER_CHAIN" \
  --pack_input \
  --pack_separated \
  --packstat

date
EOF
)

    echo "$job_submit_output"
    job_id=$(echo "$job_submit_output" | sed -n 's/Job <\([0-9]\+\)>.*/\1/p')
    if [[ -n "$job_id" ]]; then
        JOB_IDS+=("$job_id")
    fi
done

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    DEP=$(printf "done(%s) && " "${JOB_IDS[@]}")
    DEP=${DEP% && }

    MERGE_SCRIPT="$HOME/Exercise_Phase2_design/merge_pyrosetta_csvs.py"
    MERGED_CSV="$MERGED_DIR/pyrosetta_scores_merged.csv"

    bsub -w "$DEP" <<EOF
#!/bin/bash
#BSUB -J pyro_merge
#BSUB -q ${QUEUE}
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -o ${LOG_DIR}/merge.%J.out
#BSUB -e ${LOG_DIR}/merge.%J.err

set -euo pipefail
date

PYTHON_BIN="${PYTHON_BIN}"
MERGE_SCRIPT="${MERGE_SCRIPT}"
SHARD_RESULT_ROOT="${SHARD_RESULT_ROOT}"
MERGED_CSV="${MERGED_CSV}"

"\$PYTHON_BIN" "\$MERGE_SCRIPT" \
  --shard_root "\$SHARD_RESULT_ROOT" \
  --out_csv "\$MERGED_CSV"

date
EOF
fi

echo
echo "Submitted ${#JOB_IDS[@]} PyRosetta shard jobs."
echo "Shard inputs:  $SHARD_INPUT_ROOT"
echo "Shard outputs: $SHARD_RESULT_ROOT"
echo "Merged csv:    $MERGED_DIR/pyrosetta_scores_merged.csv"