#!/bin/bash
# Table VI under ONE protocol.
#
# The previous tree mixed runs: ctc_grammar's timing came from an earlier job
# with different flags, so its 1556 ms was not comparable to the 1269 ms
# measured for ctc_lexicon. Every row here is decoded alone, on one GPU, with
# identical flags, in one job.
#SBATCH --job-name=ssemg-lat-p2
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

# Frozen tuned values, identical for every row.
COMMON="--normalizer artifacts/emg_norm.pkl --lexicon artifacts/lexicon.txt \
        --train_dir_for_lm data/train_emg --beta 0.0 --kappa 0.0 --gamma 0.45 \
        --alpha 0.0 --delta 5.0 --min_eos_len 2 --beam 48 --max_len 64 \
        --ctc_grammar_beam 32 --ctc_grammar_pool 8 --lambda_ar 0.5 \
        --lambda_fix 2.0 --lambda0 1.9 --a 0.0 --b 0.0 \
        --lambda_min 0.0 --lambda_max 2.0 --ctc_select_lengthnorm"

rm -rf "$OUT/latency"; mkdir -p "$OUT/latency"
for dec in ctc_greedy ctc_lexicon ctc_grammar ctc_grammar_ar ar_greedy ar_beam \
           ns ns_ctc_select ns_joint_rerank ns_joint_adaptive; do
  echo "== latency: $dec"
  python inference_emg_llm.py --ckpt "$CK" --data_dir data/test_emg \
    --out "$OUT/latency/${dec}.jsonl" --decoder "$dec" --measure_latency \
    $COMMON > "logs/lat_${dec}.log" 2>&1 \
    && echo "  ok  $dec" || echo "  FAIL $dec (logs/lat_${dec}.log)"
done

echo
echo "================ Table VI (test, single protocol) ================"
python scripts/evaluate.py "$OUT"/latency/*.jsonl --csv "$OUT/table6_latency.csv"

echo
echo "================ Fig. 3a validity (test) ================"
python scripts/analyze_pool.py "$OUT/latency" --csv "$OUT/fig3_validity.csv" \
  --channels ar_beam ctc_greedy ctc_lexicon ctc_grammar ctc_grammar_ar
