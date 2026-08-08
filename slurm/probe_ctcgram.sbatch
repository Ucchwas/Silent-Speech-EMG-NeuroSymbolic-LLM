#!/bin/bash
# Grammar-constrained CTC decoding vs plain CTC greedy, every backbone, validation.
#
# Tests whether applying the symbolic constraint DURING CTC search beats both
# the unconstrained channel and post-hoc nearest-valid-string repair.
#SBATCH --job-name=ssemg-probe-ctcgram
#SBATCH --partition=h100
#SBATCH --account=morshedlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
banner
OUT=artifacts/probe_ctcgram
mkdir -p "$OUT"

for tag in llama32_1b llama32_3b qwen25_3b_instruct mistral7b_v03; do
  ck="artifacts/best_checkpoint_${tag}.pt"
  [ -f "$ck" ] || { echo "WARN: no checkpoint $tag"; continue; }
  python inference_emg_llm.py --ckpt "$ck" --data_dir data/val_emg \
    --out "$OUT/${tag}.jsonl" --decoder ctc_grammar \
    --ctc_grammar_beam 32 --beta 0.0 --max_len 64 \
    --normalizer artifacts/emg_norm.pkl --lexicon artifacts/lexicon.txt \
    --train_dir_for_lm data/train_emg > "logs/ctcgram_${tag}.log" 2>&1 \
    && echo "  ok  $tag" || echo "  FAIL $tag (logs/ctcgram_${tag}.log)"
done

echo
echo "================ grammar-constrained CTC (validation) ================"
python scripts/evaluate.py "$OUT"/*.jsonl --csv "$OUT/ctcgram.csv"
echo
echo "---- per-channel within each file ----"
for f in "$OUT"/*.jsonl; do
  echo "== $(basename "$f" .jsonl)"
  for col in ctc_greedy ctc_grammar ar_beam; do
    python scripts/evaluate.py "$f" --field "$col" 2>/dev/null | tail -1 | sed "s/^/    $col  /"
  done
done
