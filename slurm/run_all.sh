#!/bin/bash
# Submit the whole pipeline as one dependency graph.
#
#   bash slurm/run_all.sh              # 4 GPUs per training job
#   NGPU=2 bash slurm/run_all.sh       # 2 GPUs
#   NGPU=1 bash slurm/run_all.sh       # 1 GPU, for a busy cluster
#   EPOCHS=400 NGPU=4 bash slurm/run_all.sh
#
# Graph:
#
#   train llama32_1b ───────→ tune llama32_1b ──┐
#   train llama32_3b ───────→ tune llama32_3b ──┤
#   train qwen25_3b  ───────→ tune qwen25_3b  ──┼→ decode + score
#   train mistral7b  ───────→ tune mistral7b  ──┤
#   train sup_ar_only ──────────────────────────┤   (Table III rows reuse the
#   train sup_ctc_only ─────────────────────────┘    primary's tuned recipe)
#
# Every edge is `afterok`, so a failed training run stops its dependents rather
# than letting decode score a stale or missing checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs artifacts

NGPU=${NGPU:-4}
EPOCHS=${EPOCHS:-800}
EVAL_EVERY=${EVAL_EVERY:-10}
PRIMARY_TAG=${PRIMARY_TAG:-llama32_1b}

# Scale CPUs and memory with the GPU count; 8 CPU / 64G per GPU.
CPUS=$(( NGPU * 8 ))
MEM=$(( NGPU * 64 ))G
RES="--gres=gpu:${NGPU} --cpus-per-task=${CPUS} --mem=${MEM}"

HF=~/.cache/huggingface/hub
declare -A BACKBONES=(
  [llama32_1b]="$HF/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
  [llama32_3b]="models/llama3.2-3B"
  [qwen25_3b_instruct]="$HF/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [mistral7b_v03]="$HF/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
)

for tag in "${!BACKBONES[@]}"; do
  [ -d "${BACKBONES[$tag]}" ] || { echo "FATAL: backbone missing for $tag: ${BACKBONES[$tag]}" >&2; exit 1; }
done

echo "NGPU=$NGPU  EPOCHS=$EPOCHS  EVAL_EVERY=$EVAL_EVERY  primary=$PRIMARY_TAG"
echo

# A checkpoint that already exists is reused unless FORCE_TRAIN=1. Retraining a
# backbone takes hours, so this matters when only part of the pipeline needs to
# be redone; delete the .pt (or set FORCE_TRAIN=1) to force a fresh run.
FORCE_TRAIN=${FORCE_TRAIN:-0}

# Writes ONLY the job id to stdout -- it is captured with $(...), so every
# human-readable line here must go to stderr or it becomes the "job id".
submit_train() {  # $1 tag, then extra KEY=VALUE pairs for --export
  local tag="$1"; shift
  local ck="artifacts/best_checkpoint_${tag}.pt"
  if [ "$FORCE_TRAIN" != "1" ] && [ -f "$ck" ]; then
    printf "  %-20s SKIP (checkpoint exists; FORCE_TRAIN=1 to retrain)\n" "$tag" >&2
    return 0
  fi
  local extra=""; [ "$#" -gt 0 ] && extra=",$(IFS=,; echo "$*")"
  sbatch --parsable $RES --job-name="ssemg-train-$tag" \
    --export=ALL,TAG="$tag",BASE="$BASE_FOR_TAG",EPOCHS="$EPOCHS",EVAL_EVERY="$EVAL_EVERY"$extra \
    slurm/train.sbatch
}

echo "=== stage 1: train the four backbones ==="
declare -A TRAIN_ID
for tag in llama32_1b llama32_3b qwen25_3b_instruct mistral7b_v03; do
  BASE_FOR_TAG="${BACKBONES[$tag]}"
  jid=$(submit_train "$tag")
  if [ -n "${jid// /}" ]; then TRAIN_ID[$tag]=$jid; printf "  %-20s %s\n" "$tag" "$jid"; fi
done

echo
echo "=== stage 2: training-supervision ablation (Table III) ==="
# AR-only means the CTC objective is switched off (lambda_ctc = 0); CTC-only
# disables the AR objective and is selected on CTC greedy inside the trainer.
SUP_IDS=()
BASE_FOR_TAG="${BACKBONES[$PRIMARY_TAG]}"
# Tagged with the primary: Table III is only a controlled comparison if all
# three rows share the backbone that Tables II/IV-VI use, so changing
# PRIMARY_TAG must retrain these rather than reuse another backbone's.
for sup in ar_only ctc_only; do
  # Written as if/then, not `[ ... ] && ...`: under `set -e` a false test as
  # the last command of an AND-list exits the script, which would silently
  # drop the ctc_only job.
  extra=(SUPERVISION="$sup")
  if [ "$sup" = "ar_only" ]; then extra+=(LAMBDA_CTC=0.0); fi
  jid=$(submit_train "sup_${sup}_${PRIMARY_TAG}" "${extra[@]}")
  if [ -n "${jid// /}" ]; then SUP_IDS+=("$jid"); printf "  %-20s %s\n" "sup_${sup}" "$jid"; fi
done

echo
echo "=== stage 3: per-backbone NS tuning on validation ==="
# The decoding optimum is not shared across backbones -- on Llama-3.2-3B the
# joint AR+CTC score saturates at K=12 while Llama-3.2-1B keeps improving to
# K=48 -- so each backbone is tuned with the identical protocol on its own
# validation pools. Tuning is sequential per utterance, hence 1 GPU each.
TUNE_IDS=()
for tag in llama32_1b llama32_3b qwen25_3b_instruct mistral7b_v03; do
  # Only wait on training if we actually submitted it; a reused checkpoint is
  # already on disk, so the tune can start immediately.
  dep=""
  [ -n "${TRAIN_ID[$tag]:-}" ] && dep="--dependency=afterok:${TRAIN_ID[$tag]}"
  jid=$(sbatch --parsable --gres=gpu:1 --cpus-per-task=8 --mem=64G \
        --job-name="ssemg-tune-$tag" $dep \
        --export=ALL,CK="artifacts/best_checkpoint_${tag}.pt",OUT="artifacts/ns_tuning_${tag}.json" \
        slurm/tune.sbatch)
  TUNE_IDS+=("$jid")
  printf "  %-20s %s  (after %s)\n" "$tag" "$jid" "${TRAIN_ID[$tag]:-none}"
done

echo
echo "=== stage 4: decode + score ==="
# Concatenate only the jobs that exist; an empty element would produce
# "afterok:123::456", which SLURM rejects.
ALL_DEPS=("${TUNE_IDS[@]}" "${SUP_IDS[@]}")
DEPS=$(IFS=:; echo "${ALL_DEPS[*]}")
DEC=$(sbatch --parsable $RES --job-name=ssemg-decode \
      --dependency=afterok:"$DEPS" \
      --export=ALL,PRIMARY_TAG="$PRIMARY_TAG" slurm/decode.sbatch)
echo "  decode $DEC  (after $DEPS)"

echo
echo "watch:  squeue -u \$USER"
echo "results: Results_reproduced/"
