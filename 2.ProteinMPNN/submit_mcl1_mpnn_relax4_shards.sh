#!/bin/bash
set -euo pipefail

# ===== 你通常只改这两项 =====
QUEUE="${QUEUE:-33}"          # 之后你自己换队列名即可
N_SHARDS="${N_SHARDS:-20}"    # 默认切成20份；想更少作业就改成10
# ===========================

MICROMAMBA="$HOME/bin/micromamba"
ENV_NAME="proteinmpnn_binder_design"
SCRIPT="$HOME/dl_binder_design/mpnn_fr/dl_interface_design.py"

INPUT_DIR="/scratch/2026-03-20/bme-yaozm/mcl1_test/example_outputs"
OUT_DIR="/scratch/2026-03-20/bme-yaozm/mcl1_test/mpnn_relax4_out"

RUNLIST_DIR="$OUT_DIR/runlists"
CHECKPOINT_DIR="$OUT_DIR/checkpoints"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$RUNLIST_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

ALL_TAGS="$RUNLIST_DIR/all_tags.txt"

# 1) 生成全部tag列表（不改原文件结构）
python3 - <<'PY' "$INPUT_DIR" "$ALL_TAGS"
import sys, os, glob
input_dir, out_file = sys.argv[1], sys.argv[2]
tags = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(input_dir, "*.pdb"))
)
if not tags:
    raise SystemExit(f"No pdb files found in {input_dir}")
with open(out_file, "w") as f:
    for t in tags:
        f.write(t + "\n")
print(f"Wrote {len(tags)} tags to {out_file}")
PY

# 2) 按 N_SHARDS 均匀切分 runlist
python3 - <<'PY' "$ALL_TAGS" "$RUNLIST_DIR" "$N_SHARDS"
import sys, os, math
all_tags_file, runlist_dir, n_shards = sys.argv[1], sys.argv[2], int(sys.argv[3])

with open(all_tags_file) as f:
    tags = [x.strip() for x in f if x.strip()]

n = len(tags)
chunk = math.ceil(n / n_shards)

# 先清理旧 runlist
for fn in os.listdir(runlist_dir):
    if fn.startswith("runlist_") and fn.endswith(".txt"):
        os.remove(os.path.join(runlist_dir, fn))

for i in range(n_shards):
    sub = tags[i*chunk:(i+1)*chunk]
    if not sub:
        continue
    fn = os.path.join(runlist_dir, f"runlist_{i:02d}.txt")
    with open(fn, "w") as f:
        for t in sub:
            f.write(t + "\n")
    print(f"{fn}: {len(sub)} tags")
PY

# 3) 逐个 runlist 提交小作业
for RUNLIST in "$RUNLIST_DIR"/runlist_*.txt; do
    shard=$(basename "$RUNLIST" .txt)
    CHECKPOINT="$CHECKPOINT_DIR/${shard}.check.point"

    bsub <<EOF
#!/bin/bash
#BSUB -J ${shard}
#BSUB -q ${QUEUE}
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -o ${LOG_DIR}/${shard}.%J.out
#BSUB -e ${LOG_DIR}/${shard}.%J.err

set -euo pipefail
date

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MICROMAMBA="$MICROMAMBA"
ENV_NAME="$ENV_NAME"
SCRIPT="$SCRIPT"
INPUT_DIR="$INPUT_DIR"
OUT_DIR="$OUT_DIR"
RUNLIST="$RUNLIST"
CHECKPOINT="$CHECKPOINT"

rm -f "$CHECKPOINT"

\$MICROMAMBA run -n "\$ENV_NAME" python "\$SCRIPT" \
  -pdbdir "\$INPUT_DIR" \
  -outpdbdir "\$OUT_DIR" \
  -runlist "\$RUNLIST" \
  -checkpoint_name "\$CHECKPOINT" \
  -relax_cycles 4 \
  -seqs_per_struct 1 \
  -output_intermediates

date
EOF

done

echo
echo "Submitted all shards from $RUNLIST_DIR"