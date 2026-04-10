#!/bin/bash
set -euo pipefail

# ===== 直接照你原始 lsf 填 =====
QUEUE="8v100-32-sc"

MICROMAMBA="$HOME/bin/micromamba"
ENV_NAME="SE3nv"
RFDIFFUSION_DIR="$HOME/RFdiffusion"

WORKDIR="/scratch/2026-03-28/bme-yaozm/mcl1_test"
INPUT_PDB="$HOME/Exercise_Phase2_design/mcl1_test/MCL1_A_noLigand_filled.pdb"
OUTPUT_PREFIX="$WORKDIR/example_outputs/diffused_binder_cyclic_mcl1"

N_SHARDS=10
DESIGNS_PER_SHARD=50
# ==============================

mkdir -p "$WORKDIR"
cd "$WORKDIR"

for i in $(seq 0 $((N_SHARDS-1))); do
    shard=$(printf "%02d" "$i")
    shard_prefix="${OUTPUT_PREFIX}_shard${shard}"

    bsub <<EOF
#!/bin/bash
#BSUB -J mcl1_rfd_${shard}
#BSUB -q ${QUEUE}
#BSUB -n 8
#BSUB -R "span[ptile=8]"
#BSUB -gpu "num=1/host"
#BSUB -o ${WORKDIR}/mcl1_rfd_${shard}.%J.out
#BSUB -e ${WORKDIR}/mcl1_rfd_${shard}.%J.err

set -euo pipefail
date

module load cuda/11.8

MICROMAMBA="${MICROMAMBA}"
ENV_NAME="${ENV_NAME}"
RFDIFFUSION_DIR="${RFDIFFUSION_DIR}"
INPUT_PDB="${INPUT_PDB}"
SHARD_PREFIX="${shard_prefix}"

export HYDRA_FULL_ERROR=1

\$MICROMAMBA run -n "\$ENV_NAME" python "\$RFDIFFUSION_DIR/scripts/run_inference.py" \
  --config-name base \
  inference.output_prefix="\$SHARD_PREFIX" \
  inference.num_designs=${DESIGNS_PER_SHARD} \
  'contigmap.contigs=[16-16 A1-145/0]' \
  inference.input_pdb="\$INPUT_PDB" \
  inference.cyclic=True \
  diffuser.T=50 \
  inference.cyc_chains='a'

date
EOF

done

echo "Submitted ${N_SHARDS} RFdiffusion jobs, ${DESIGNS_PER_SHARD} designs each."