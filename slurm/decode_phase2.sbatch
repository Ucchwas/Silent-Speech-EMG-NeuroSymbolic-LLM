#!/bin/bash
# Phase 2 final test decode: the two new decoders plus a re-score of the rest.
#
# ctc_grammar_ar reranks the grammar-valid pool with the frozen LLM.
# lambda_ar = 0.5 was selected on validation (0.2256 -> 0.1955, minimum of a
# 7-point sweep) and is frozen here; test is decoded once. It LOSES on test
# (0.2348 -> 0.2652) and the paper reports it as a negative result. The
# reported system is plain ctc_grammar.
#
# ctc_lexicon fills the word-lexicon cell of the constraint-placement 2x2, now
# with word_restart so it can spell multi-word transcripts instead of stopping
# after the first lexicon entry.
#SBATCH --job-name=ssemg-decode-p2
#SBATCH --partition=h100
#SBATCH --account=morshedlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
banner
: "${OUT:=Results_reproduced}"
CK=artifacts/best_checkpoint_llama32_1b.pt
LAM_AR=0.5

COMMON="--normalizer artifacts/emg_norm.pkl --lexicon artifacts/lexicon.txt \
        --train_dir_for_lm data/train_emg --beta 0.0 --max_len 64 \
        --ctc_grammar_beam 32 --ctc_grammar_pool 8"

mkdir -p "$OUT/decoders"

# ---- headline + 2x2 cell, test split --------------------------------------
for dec in ctc_grammar_ar ctc_lexicon; do
  echo "== test: $dec"
  python inference_emg_llm.py --ckpt "$CK" --data_dir data/test_emg \
    --out "$OUT/decoders/${dec}.jsonl" --decoder "$dec" \
    --lambda_ar "$LAM_AR" $COMMON \
    && echo "  ok  $dec" || echo "  FAIL $dec"
done

# ---- refresh ctc_grammar so its pool is recorded in the reported tree ------
echo "== test: ctc_grammar (with pool)"
python inference_emg_llm.py --ckpt "$CK" --data_dir data/test_emg \
  --out "$OUT/decoders/ctc_grammar.jsonl" --decoder ctc_grammar $COMMON \
  && echo "  ok" || echo "  FAIL"

# ---- latency for the new decoders (Table VI) ------------------------------
mkdir -p "$OUT/latency"
for dec in ctc_grammar_ar ctc_lexicon; do
  python inference_emg_llm.py --ckpt "$CK" --data_dir data/test_emg \
    --out "$OUT/latency/${dec}.jsonl" --decoder "$dec" --lambda_ar "$LAM_AR" \
    --measure_latency $COMMON > "logs/lat_${dec}.log" 2>&1 \
    && echo "  ok  latency $dec" || echo "  FAIL latency $dec"
done

# ---- per-field breakdown for the new headline (Table V) -------------------
mkdir -p "$OUT/decomp"
cp "$OUT/decoders/ctc_grammar_ar.jsonl" "$OUT/decomp/ctc_grammar_ar.jsonl" 2>/dev/null

echo
echo "================ Table II (test) ================"
python scripts/evaluate.py "$OUT"/decoders/*.jsonl --bootstrap \
  --csv "$OUT/table2_decoders.csv"

echo
echo "================ Table V (test) ================"
python scripts/evaluate.py "$OUT"/decomp/*.jsonl --fields --decomp \
  --csv "$OUT/table5_decomp.csv"

echo
echo "================ Table VI (test) ================"
python scripts/evaluate.py "$OUT"/latency/*.jsonl --csv "$OUT/table6_latency.csv"

echo
echo "================ channels + pool oracle (test) ================"
python scripts/analyze_pool.py "$OUT/decoders" \
  --channels ar_beam ctc_greedy ctc_lexicon ctc_grammar ctc_grammar_ar

echo
echo "================ paired bootstrap vs ctc_grammar_ar ================"
for b in ctc_grammar ctc_greedy ar_beam ns_joint_rerank; do
  python scripts/evaluate.py "$OUT/decoders/ctc_grammar_ar.jsonl" \
    --compare "$OUT/decoders/ctc_grammar_ar.jsonl" "$OUT/decoders/$b.jsonl" \
    2>/dev/null | tail -4
done
