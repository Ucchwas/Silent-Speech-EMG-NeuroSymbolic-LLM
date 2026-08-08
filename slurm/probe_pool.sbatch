#!/bin/bash
# Phase 2 probes E1/E2/E3, all inference-only on existing checkpoints.
#
# E1  Does the grammar-valid CTC candidate pool contain better strings than its
#     own top-1?  If the pool oracle is far below ctc_grammar's WER, the frozen
#     LLM has something to rank and ctc_grammar_ar is worth reporting; if not,
#     the ceiling is the result and the arbitration idea is dropped.
# E2  ctc_lexicon fills the missing cell of the constraint-placement 2x2:
#     the same CTC prefix beam under the flat word lexicon instead of the
#     utterance grammar, isolating granularity from channel.
# E3  Does the AR objective improve the CTC head?  Decode the supervision
#     variants with ctc_grammar.  sup_ar_only has no trained CTC head and is
#     deliberately absent.
#
# Validation split throughout: E1 gates a design decision and tunes lambda_ar,
# so it must not touch test.
#SBATCH --job-name=ssemg-probe-pool
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
OUT=artifacts/probe_pool
mkdir -p "$OUT"

COMMON="--normalizer artifacts/emg_norm.pkl --lexicon artifacts/lexicon.txt \
        --train_dir_for_lm data/train_emg --beta 0.0 --max_len 64 \
        --ctc_grammar_beam 32 --ctc_grammar_pool 8"

# ---- E1 + E2: primary backbone, validation --------------------------------
echo "== E1/E2: pool + lexicon cell (llama32_1b, val)"
python inference_emg_llm.py --ckpt artifacts/best_checkpoint_llama32_1b.pt \
  --data_dir data/val_emg --out "$OUT/val_1b.jsonl" \
  --decoder ctc_grammar \
  --also ctc_grammar_ar ctc_lexicon ctc_greedy ar_beam $COMMON \
  && echo "  ok" || echo "  FAIL"

# ---- E1: tune lambda_ar on validation -------------------------------------
# The pool oracle showed 60% relative headroom below the CTC top-1, so the
# question is only how much of it AR rescoring recovers. Swept here, frozen
# before test.
for lam in 0.0 0.25 0.5 1.0 2.0 4.0 8.0; do
  python inference_emg_llm.py --ckpt artifacts/best_checkpoint_llama32_1b.pt \
    --data_dir data/val_emg --out "$OUT/lamar_${lam}.jsonl" \
    --decoder ctc_grammar_ar --lambda_ar "$lam" $COMMON > /dev/null 2>&1 \
    && echo "  ok  lambda_ar=$lam" || echo "  FAIL lambda_ar=$lam"
done

# ---- E3: supervision variants decoded through the grammar -----------------
for tag in sup_ctc_only_llama32_1b llama32_1b; do
  ck="artifacts/best_checkpoint_${tag}.pt"
  [ -f "$ck" ] || { echo "WARN: no checkpoint $tag"; continue; }
  echo "== E3: $tag"
  python inference_emg_llm.py --ckpt "$ck" --data_dir data/test_emg \
    --out "$OUT/sup_${tag}.jsonl" --decoder ctc_grammar $COMMON \
    && echo "  ok  $tag" || echo "  FAIL $tag"
done

echo
echo "================ E1: pool oracle (validation) ================"
python scripts/analyze_pool.py "$OUT/val_1b.jsonl"

echo
echo "================ E1/E2: decoder comparison (validation) ================"
python scripts/analyze_pool.py "$OUT/val_1b.jsonl" | tail -20

echo
echo "================ E1: lambda_ar sweep (validation) ================"
python scripts/evaluate.py "$OUT"/lamar_*.jsonl --csv "$OUT/lambda_ar.csv"

echo
echo "================ E3: supervision x ctc_grammar (test) ================"
python scripts/evaluate.py "$OUT"/sup_*.jsonl --csv "$OUT/e3_supervision.csv"
