# Shared environment for every job. Sourced, never executed directly.
#
# Two things here are not optional and have bitten this pipeline before:
#   * PYTHONPATH must include the repo root. torchrun does not put the launch
#     directory on sys.path, so `import train_emg_llm` fails without it.
#   * HF_HUB_OFFLINE=1. All four backbones are already in the HF cache; without
#     this a compute node with no route to the internet stalls instead of
#     failing fast.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/miniforge3/etc/profile.d/conda.sh
set +u; conda activate ss2; set -u

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4          # unpinned, torch spawns ~1 thread/core and thrashes
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# Number of GPUs actually allocated to this job, whatever --gres asked for.
detect_ngpu() {
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    echo "$SLURM_GPUS_ON_NODE"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
  else
    echo 1
  fi
}

banner() {
  echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  gpus=$(detect_ngpu)"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
  echo "-----------------------------------------------------------------"
}
