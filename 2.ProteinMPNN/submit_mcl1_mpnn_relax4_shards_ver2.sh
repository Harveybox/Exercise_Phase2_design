#!/bin/bash
set -euo pipefail

# ===== 你通常只改这两项 =====
QUEUE="${QUEUE:-33}"          # 之后你自己换队列名即可
N_SHARDS="${N_SHARDS:-20}"    # 默认切成20份
# ===========================

MICROMAMBA="$HOME/bin/micromamba"
ENV_NAME="proteinmpnn_binder_design"
SCRIPT="$HOME/dl_binder_design/mpnn_fr/dl_interface_design.py"

# 获取 Python 绝对路径，避开 mamba 锁
PYTHON_BIN="$($MICROMAMBA run -n "$ENV_NAME" which python)"

INPUT_DIR="/scratch/2026-03-28/bme-yaozm/mcl1_test/example_outputs"
OUT_DIR="/scratch/2026-03-28/bme-yaozm/mcl1_test/mpnn_relax4_out"

RUNLIST_DIR="$OUT_DIR/runlists"
CHECKPOINT_DIR="$OUT_DIR/checkpoints"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$RUNLIST_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

ALL_TAGS="$RUNLIST_DIR/all_tags.txt"

# 1) 生成全部tag列表
python3 - <<'PY' "$INPUT_DIR" "$ALL_TAGS"
import sys, os, glob
input_dir, out_file = sys.argv[1], sys.argv[2]
tags = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(input_dir, "*.pdb"))
)
with open(out_file, "w") as f:
    for t in tags:
        f.write(t + "\n")
PY

# 2) 切分任务
python3 - <<'PY' "$ALL_TAGS" "$N_SHARDS" "$RUNLIST_DIR"
import sys, os, math
all_tags_file, n_shards, runlist_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(all_tags_file) as f:
    tags = [x.strip() for x in f if x.strip()]

n = len(tags)
chunk = math.ceil(n / n_shards)

for i in range(n_shards):
    sub = tags[i*chunk:(i+1)*chunk]
    if not sub:
        continue
    fn = os.path.join(runlist_dir, f"runlist_{i:02d}.txt")
    with open(fn, "w") as f:
        for t in sub:
            f.write(t + "\n")
PY

# 3) 逐个 runlist 提交小作业
for RUNLIST in "$RUNLIST_DIR"/runlist_*.txt; do
    shard=$(basename "$RUNLIST" .txt)
    CHECKPOINT="$CHECKPOINT_DIR/${shard}.check.point"

    bsub <<EOF
#!/bin/bash
#BSUB -J mpnn_${shard}
#BSUB -q ${QUEUE}
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -R "rusage[mem=2000]"
#BSUB -o ${LOG_DIR}/${shard}.%J.out
#BSUB -e ${LOG_DIR}/${shard}.%J.err

set -euo pipefail
date

# 环境变量设置
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 将主脚本变量注入计算节点
PYTHON_BIN="${PYTHON_BIN}"
SCRIPT="${SCRIPT}"
INPUT_DIR="${INPUT_DIR}"
OUT_DIR="${OUT_DIR}"
RUNLIST="${RUNLIST}"
CHECKPOINT="${CHECKPOINT}"

# 清理旧的 checkpoint，防止中断重跑时发生读写冲突
rm -f "\$CHECKPOINT"

# 执行行，全部使用局部的计算节点变量
\$PYTHON_BIN "\$SCRIPT" \
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
echo "Submitted ${N_SHARDS} shards to ${QUEUE}."