#!/bin/bash
# Tune the NS decoding hyperparameters for ONE checkpoint, on VALIDATION only.
#
#   sbatch --export=ALL,CK=artifacts/best_checkpoint_llama32_3b.pt,\
#OUT=artifacts/ns_tuning_llama32_3b.json slurm/tune.sbatch
#
# Search is inherently sequential over utterances, so this uses a single GPU
# regardless of what is allocated; parallelism comes from running one of these
# per backbone at the same time.
#
# Variables:
#   CK   checkpoint to tune for            (required)
#   OUT  where to write the tuned config   (required)
#
#SBATCH --job-name=ssemg-tune
#SBATCH --partition=h100
#SBATCH --account=morshedlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

: "${CK:?set CK}"
: "${OUT:?set OUT}"

banner
echo "tuning $CK -> $OUT"
[ -f "$CK" ] || { echo "checkpoint not found: $CK" >&2; exit 1; }

python -m scripts.tune_ns --ckpt "$CK" --data_dir data/val_emg --out "$OUT"

echo "done -> $OUT"
