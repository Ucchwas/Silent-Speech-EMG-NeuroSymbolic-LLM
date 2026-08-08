#!/bin/bash
# Val-set AR-beam vs CTC-greedy for every trained backbone.
#
# The premise of the paper is that routing EMG through a frozen LLM beats the
# adapter's own CTC head. That comparison is not stored anywhere: the trainer
# only checkpoints best_val_wer (AR beam), so CTC greedy has to be re-decoded.
#SBATCH --job-name=ssemg-probe-arctc
#SBATCH --partition=h100
#SBATCH --account=morshedlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
banner

OUT=artifacts/probe_arctc
mkdir -p "$OUT"

for tag in llama32_1b llama32_3b qwen25_3b_instruct mistral7b_v03; do
  ck="artifacts/best_checkpoint_${tag}.pt"
  [ -f "$ck" ] || { echo "WARN: no checkpoint for $tag"; continue; }
  for dec in ctc_greedy ar_beam; do
    python inference_emg_llm.py --ckpt "$ck" --data_dir data/val_emg \
      --out "$OUT/${tag}__${dec}.jsonl" --decoder "$dec" \
      --beam 6 --max_len 64 \
      --normalizer artifacts/emg_norm.pkl \
      --lexicon artifacts/lexicon.txt \
      --train_dir_for_lm data/train_emg > "logs/probe_${tag}_${dec}.log" 2>&1 \
      && echo "  ok  $tag $dec" || echo "  FAIL $tag $dec (see logs/probe_${tag}_${dec}.log)"
  done
done

echo
echo "================ AR beam vs CTC greedy (validation) ================"
python scripts/evaluate.py "$OUT"/*.jsonl --csv "$OUT/probe_arctc.csv"
