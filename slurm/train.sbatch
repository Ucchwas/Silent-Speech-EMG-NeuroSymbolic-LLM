#!/bin/bash
# Train the EMG adapter for ONE configuration. The LLM stays frozen throughout.
#
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=256G \
#          --export=ALL,TAG=llama32_3b,BASE=models/llama3.2-3B slurm/train.sbatch
#
# Resource flags are passed on the command line, not baked in: #SBATCH
# directives are parsed before any shell runs, so they cannot read $NGPU. The
# directives below are only defaults for a bare `sbatch slurm/train.sbatch`.
#
# Variables:
#   TAG          checkpoint name -> artifacts/best_checkpoint_<TAG>.pt   (required)
#   BASE         path to the frozen backbone                            (required)
#   EPOCHS       default 800
#   EVAL_EVERY   default 10
#   SUPERVISION  ar_ctc | ar_only | ctc_only   (Table III)
#   LAMBDA_CTC   default 0.2; set 0.0 for AR-only
#   EARLY_STOP   epochs without improvement before stopping, default 200
#
#SBATCH --job-name=ssemg-train
#SBATCH --partition=h100
#SBATCH --account=morshedlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

: "${TAG:?set TAG}"
: "${BASE:?set BASE}"
: "${EPOCHS:=800}"
: "${EVAL_EVERY:=10}"
: "${SUPERVISION:=ar_ctc}"
: "${LAMBDA_CTC:=0.2}"
: "${EARLY_STOP:=200}"

NGPU=$(detect_ngpu)
banner
echo "tag=$TAG backbone=$BASE epochs=$EPOCHS supervision=$SUPERVISION lambda_ctc=$LAMBDA_CTC ngpu=$NGPU"
[ -d "$BASE" ] || { echo "backbone not found: $BASE" >&2; exit 1; }

# DDP shards the training set and, more importantly, the validation decode:
# beam search is batch-of-one and sequential (~2.4 s/utterance against ~3.5 s
# for a whole training epoch), so validation dominates wall-clock. Per-rank
# batch size is batch_size/world inside train_emg_llm.py, which keeps the
# EFFECTIVE batch at 8 and the optimizer-step count per epoch unchanged, so the
# recipe is identical at 1, 2 or 4 GPUs.
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train_emg_llm.py \
  --base_model "$BASE" \
  --tag "$TAG" \
  --train_dir data/train_emg \
  --val_dir data/val_emg \
  --artifacts_dir artifacts \
  --epochs "$EPOCHS" \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --lambda_ctc "$LAMBDA_CTC" \
  --supervision "$SUPERVISION" \
  --augment \
  --aug_strength 1.0 \
  --dropout 0.1 \
  --label_smoothing 0.0 \
  --adapter_norm layer \
  --soft_prompt_len 8 \
  --inst_prompt_len 8 \
  --subsample_factor 4 \
  --beam_size 6 \
  --max_decode_len 128 \
  --ar_len_alpha 0.6 \
  --ar_len_delta 5.0 \
  --eval_every "$EVAL_EVERY" \
  --early_stop "$EARLY_STOP" \
  --num_workers 4 \
  --seed 1234

echo "done -> artifacts/best_checkpoint_${TAG}.pt"
