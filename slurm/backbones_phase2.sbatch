#!/bin/bash
# Fig. 2 under the REPORTED decoder.
# The backbone comparison previously used ns_joint_adaptive, which is no longer
# the system the paper reports. Swapping only the frozen weights is only
# informative if the decoder is the one in Table II.
#SBATCH --job-name=ssemg-bb-p2
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
OUT=Results_reproduced/backbones
rm -rf "$OUT"; mkdir -p "$OUT"
COMMON="--normalizer artifacts/emg_norm.pkl --lexicon artifacts/lexicon.txt \
        --train_dir_for_lm data/train_emg --beta 0.0 --max_len 64 \
        --ctc_grammar_beam 32 --ctc_grammar_pool 8"
for tag in llama32_1b llama32_3b qwen25_3b_instruct mistral7b_v03; do
  ck="artifacts/best_checkpoint_${tag}.pt"
  [ -f "$ck" ] || { echo "WARN no ckpt $tag"; continue; }
  python inference_emg_llm.py --ckpt "$ck" --data_dir data/test_emg \
    --out "$OUT/${tag}.jsonl" --decoder ctc_grammar $COMMON \
    > "logs/bb_${tag}.log" 2>&1 && echo "  ok $tag" || echo "  FAIL $tag"
done
echo; echo "============ Fig. 2 (test, ctc_grammar) ============"
python scripts/evaluate.py "$OUT"/*.jsonl --bootstrap --csv Results_reproduced/fig2_backbones.csv
