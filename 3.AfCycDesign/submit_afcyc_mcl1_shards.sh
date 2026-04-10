#!/bin/bash
set -euo pipefail

# ====== 按你现有 run_afcyc_mcl1.lsf 里的值填这几项 ======
QUEUE="${QUEUE:-8v100-32-sc}"                         # 你自己的队列名
N_SHARDS="${N_SHARDS:-15}"                   # 默认切 15 份
MICROMAMBA="$HOME/bin/micromamba"
ENV_NAME="afcycdesign"                       # 按你现有环境名改
SCRIPT="$HOME/Exercise_Phase2_design/afcyc_predict_batch.py"

INPUT_DIR="/scratch/2026-03-28/bme-yaozm/mcl1_test/mpnn_relax4_out"
OUT_BASE="/scratch/2026-03-28/bme-yaozm/mcl1_test/afcyc_out"
AF_PARAMS_DIR="$HOME/dl_binder_design/af2_initial_guess/model_weights/params"

TARGET_CHAIN="A"
BINDER_CHAIN="B"

# 这三项也尽量照你现有 lsf
NCPU=1
PTILE=1
GPU_REQ='num=1/host'
# ======================================================

RUNLIST_DIR="$OUT_BASE/runlists"
SHARD_INPUT_ROOT="$OUT_BASE/shard_inputs"
SHARD_RESULT_ROOT="$OUT_BASE/shard_results"
LOG_DIR="$OUT_BASE/logs"
MERGED_DIR="$OUT_BASE/merged"

mkdir -p "$RUNLIST_DIR" "$SHARD_INPUT_ROOT" "$SHARD_RESULT_ROOT" "$LOG_DIR" "$MERGED_DIR"

ALL_TAGS="$RUNLIST_DIR/all_tags.txt"

# 1) 收集所有 relaxed pdb 的 basename（不改原目录结构）
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

# 2) 均匀切成 N_SHARDS 份
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

# 3) 为每个 shard 建一个只含符号链接的 input_dir
for RUNLIST in "$RUNLIST_DIR"/runlist_*.txt; do
    shard=$(basename "$RUNLIST" .txt)
    SHARD_INPUT_DIR="$SHARD_INPUT_ROOT/$shard"
    mkdir -p "$SHARD_INPUT_DIR"
    find "$SHARD_INPUT_DIR" -maxdepth 1 -type l -name "*.pdb" -delete

    while IFS= read -r pdb_name; do
        ln -sf "$INPUT_DIR/$pdb_name" "$SHARD_INPUT_DIR/$pdb_name"
    done < "$RUNLIST"
done

# 4) 提交每个 shard
JOB_IDS=()

for RUNLIST in "$RUNLIST_DIR"/runlist_*.txt; do
    shard=$(basename "$RUNLIST" .txt)
    SHARD_INPUT_DIR="$SHARD_INPUT_ROOT/$shard"
    SHARD_OUT_DIR="$SHARD_RESULT_ROOT/$shard"

    mkdir -p "$SHARD_OUT_DIR"

    job_submit_output=$(bsub <<EOF
#!/bin/bash
#BSUB -J afcyc_${shard}
#BSUB -q ${QUEUE}
#BSUB -n ${NCPU}
#BSUB -R "span[ptile=${PTILE}]"
#BSUB -gpu "${GPU_REQ}"
#BSUB -o ${LOG_DIR}/${shard}.%J.out
#BSUB -e ${LOG_DIR}/${shard}.%J.err

set -euo pipefail
date

MICROMAMBA="${MICROMAMBA}"
ENV_NAME="${ENV_NAME}"
SCRIPT="${SCRIPT}"
SHARD_INPUT_DIR="${SHARD_INPUT_DIR}"
SHARD_OUT_DIR="${SHARD_OUT_DIR}"
AF_PARAMS_DIR="${AF_PARAMS_DIR}"
TARGET_CHAIN="${TARGET_CHAIN}"
BINDER_CHAIN="${BINDER_CHAIN}"

mkdir -p "\$SHARD_OUT_DIR"

\$MICROMAMBA run -n "\$ENV_NAME" python "\$SCRIPT" \
  --input_dir "\$SHARD_INPUT_DIR" \
  --out_dir "\$SHARD_OUT_DIR" \
  --data_dir "\$AF_PARAMS_DIR" \
  --target_chain "\$TARGET_CHAIN" \
  --binder_chain "\$BINDER_CHAIN" \
  --offset_type 2 \
  --num_recycles 3 \
  --num_models 1

date
EOF
)

    echo "$job_submit_output"
    job_id=$(echo "$job_submit_output" | sed -n 's/Job <\([0-9]\+\)>.*/\1/p')
    if [[ -n "$job_id" ]]; then
        JOB_IDS+=("$job_id")
    fi
done

# 5) 自动提交 merge 任务（等所有 shard 完成后）
if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    DEP=$(printf "done(%s) && " "${JOB_IDS[@]}")
    DEP=${DEP% && }

    MERGE_SCRIPT="$HOME/Exercise_Phase2_design/merge_afcyc_csvs.py"
    MERGED_CSV="$MERGED_DIR/results_merged.csv"

    bsub -w "$DEP" <<EOF
#!/bin/bash
#BSUB -J afcyc_merge
#BSUB -q 33
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -o ${LOG_DIR}/merge.%J.out
#BSUB -e ${LOG_DIR}/merge.%J.err

set -euo pipefail
date

MICROMAMBA="${MICROMAMBA}"
ENV_NAME="${ENV_NAME}"
MERGE_SCRIPT="${MERGE_SCRIPT}"
SHARD_RESULT_ROOT="${SHARD_RESULT_ROOT}"
MERGED_CSV="${MERGED_CSV}"

\$MICROMAMBA run -n "\$ENV_NAME" python "\$MERGE_SCRIPT" \
  --shard_root "\$SHARD_RESULT_ROOT" \
  --out_csv "\$MERGED_CSV"

date
EOF
fi

echo
echo "Submitted ${#JOB_IDS[@]} AfCyc shard jobs."
echo "Shard inputs:  $SHARD_INPUT_ROOT"
echo "Shard outputs: $SHARD_RESULT_ROOT"
echo "Merged csv:    $MERGED_DIR/results_merged.csv"