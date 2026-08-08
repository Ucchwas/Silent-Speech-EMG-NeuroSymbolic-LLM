#!/bin/bash
# Decode every suite in the paper on the test split and score it into tables.
#
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=256G slurm/decode.sbatch
#
# Suites are independent decode runs, so they are fanned out NGPU at a time.
# `latency` is excluded from that fan-out and run alone afterwards: Table VI
# reports ms/utterance and peak GPU memory, and sharing a device would corrupt
# both numbers.
#
# Variables:
#   PRIMARY_TAG  backbone for Tables II-VI, default llama32_3b
#   OUT          output root, default Results_reproduced
#
#SBATCH --job-name=ssemg-decode
#SBATCH --partition=h100
#SBATCH --account=morshedlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

: "${PRIMARY_TAG:=llama32_3b}"
: "${OUT:=Results_reproduced}"
NGPU=$(detect_ngpu)
banner

PRIMARY="artifacts/best_checkpoint_${PRIMARY_TAG}.pt"
[ -f "$PRIMARY" ] || { echo "missing primary checkpoint $PRIMARY" >&2; exit 1; }

# Start from an empty tree. Overwriting in place is not enough: if a suite
# fails, its files from a previous run survive and the scoring step below
# reports them as current, silently mixing two models into one table.
if [ -d "$OUT" ]; then
  ARCHIVE="${OUT}.superseded.$(date +%Y%m%d-%H%M%S)"
  echo "archiving previous results -> $ARCHIVE"
  mv "$OUT" "$ARCHIVE"
fi
mkdir -p "$OUT" logs

# run_experiments.py reads the primary's tuned decoding config from this path.
TUNED_PRIMARY="artifacts/ns_tuning_${PRIMARY_TAG}.json"
if [ -f "$TUNED_PRIMARY" ]; then
  cp "$TUNED_PRIMARY" artifacts/ns_tuning.json
  echo "primary decoding config: $TUNED_PRIMARY"
else
  echo "WARNING: $TUNED_PRIMARY missing; falling back to Table I defaults" >&2
fi

# Lexicon and char 5-gram both come from TRAIN transcripts only.
python scripts/make_lexicon.py --src data/train_emg --out artifacts/lexicon.txt --expand_numerals

# --- smoke test --------------------------------------------------------------
# Two utterances through every decoder before committing to the full run. A
# previous pass spent 90 minutes decoding and then died in the last suite on a
# json.dump of a numpy float32 -- a failure that shows up on utterance one.
# Anything that breaks serialisation, the trie, or a decoder surfaces here in
# seconds instead of after the GPU hours are already spent.
echo "=== smoke test: 2 utterances, every decoder ==="
SMOKE=$(mktemp -d)
for dec in ctc_greedy ctc_grammar ar_beam ns ns_joint_adaptive; do
  if python inference_emg_llm.py --ckpt "$PRIMARY" --data_dir data/val_emg --limit 2 \
       --decoder "$dec" --out "$SMOKE/$dec.jsonl" \
       --normalizer artifacts/emg_norm.pkl --lexicon artifacts/lexicon.txt \
       --train_dir_for_lm data/train_emg > "$SMOKE/$dec.log" 2>&1 \
     && python -c "import json,sys
rows=[json.loads(l) for l in open('$SMOKE/$dec.jsonl') if l.strip()]
assert rows and all(r.get('final') is not None for r in rows), 'no output'
json.dumps(rows)" 2>>"$SMOKE/$dec.log"
  then
    printf "  %-18s ok\n" "$dec"
  else
    echo "  $dec FAILED -- aborting before the full run" >&2
    tail -20 "$SMOKE/$dec.log" >&2
    rm -rf "$SMOKE"; exit 1
  fi
done
rm -rf "$SMOKE"
echo

RE="python scripts/run_experiments.py --out_root $OUT"

# --- assemble the suite list -------------------------------------------------
SUITES=()
SUITES+=("decoders|$RE --suite decoders   --ckpt $PRIMARY")
SUITES+=("ablation|$RE --suite ablation   --ckpt $PRIMARY")
SUITES+=("decomp|$RE --suite decomp       --ckpt $PRIMARY")
# Fig. 4 is a multi-backbone curve, so the beta sweep gets the same checkpoint
# list as Fig. 2 (BB is assembled below, hence the deferred append further down).
SUITES+=("sweep_lam|$RE --suite sweep_lam   --ckpt $PRIMARY")

# Fig. 2: every backbone that has a checkpoint, each with its own tuned config.
BB=""
for tag in llama32_1b llama32_3b qwen25_3b_instruct mistral7b_v03; do
  ck="artifacts/best_checkpoint_${tag}.pt"
  [ -f "$ck" ] && BB="$BB --backbone_ckpt ${tag}=${ck}" || echo "WARN: no checkpoint for $tag"
done
if [ -n "$BB" ]; then
  SUITES+=("backbones|$RE --suite backbones $BB")
  SUITES+=("sweep_beta|$RE --suite sweep_beta $BB")
else
  SUITES+=("sweep_beta|$RE --suite sweep_beta --ckpt $PRIMARY")
fi

# Table III: same inference recipe, three training objectives.
SUP=""
[ -f "$PRIMARY" ] && SUP="$SUP --backbone_ckpt ar_ctc=$PRIMARY"
# Table III compares training objectives, so all three rows must share the
# backbone used by Tables II/IV-VI. The checkpoints are therefore tagged with
# the primary; an un-suffixed checkpoint from another primary is NOT accepted
# as a fallback, because silently mixing backbones into one table is exactly
# the failure this naming prevents.
for name in ar_only ctc_only; do
  ck="artifacts/best_checkpoint_sup_${name}_${PRIMARY_TAG}.pt"
  [ -f "$ck" ] && SUP="$SUP --backbone_ckpt ${name}=${ck}" \
    || echo "WARN: no sup_${name} checkpoint for primary $PRIMARY_TAG (expected $ck)"
done
[ -n "$SUP" ] && SUITES+=("supervision|$RE --suite supervision $SUP")

# --- run them, NGPU at a time ------------------------------------------------
run_batch() {
  local pids=() names=() i=0 rc=0 k=0
  for entry in "$@"; do
    local name="${entry%%|*}" cmd="${entry#*|}" g=$(( i % NGPU ))
    echo "  [$name] -> GPU $g"
    CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > "logs/suite_${name}.log" 2>&1 &
    pids+=($!); names+=("$name"); i=$((i + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then echo "SUITE FAILED: ${names[$k]} (see logs/suite_${names[$k]}.log)" >&2; rc=1; fi
    k=$((k + 1))
  done
  return $rc
}

echo "=== running ${#SUITES[@]} suites, ${NGPU} at a time ==="
fail=0
for ((s = 0; s < ${#SUITES[@]}; s += NGPU)); do
  run_batch "${SUITES[@]:s:NGPU}" || fail=1
done
for f in logs/suite_*.log; do echo "----- $f -----"; tail -5 "$f"; done
[ "$fail" -eq 0 ] || { echo "one or more suites failed; not scoring partial results" >&2; exit 1; }

# --- Table VI: latency and peak memory, exclusive access to one GPU ----------
echo "=== latency (exclusive, GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --suite latency \
    --ckpt "$PRIMARY" --out_root "$OUT"

# --- score -------------------------------------------------------------------
score() {  # $1 = label, $2 = glob dir, $3 = csv, $4.. = extra evaluate flags
  local label="$1" dir="$2" csv="$3"; shift 3
  echo; echo "################ $label ################"
  if compgen -G "$OUT/$dir/*.jsonl" > /dev/null; then
    python scripts/evaluate.py "$@" "$OUT/$dir"/*.jsonl --csv "$OUT/$csv"
  else
    echo "no outputs under $OUT/$dir"
  fi
}

score "TABLE II - decoding conditions"        decoders    table2_decoders.csv --bootstrap
score "TABLE III - training supervision"      supervision table3_supervision.csv
score "TABLE IV / FIG 3 - NS ablations"       ablation    table4_ablation.csv --bootstrap
score "TABLE V - error decomposition"         decomp      table5_decomp.csv   --decomp --decomp_words --fields
score "TABLE VI - latency and peak memory"    latency     table6_latency.csv
score "FIG 2 - backbone comparison"           backbones   fig2_backbones.csv  --bootstrap
score "FIG 4 - beta sweep (validation)"       sweep_beta  fig4_beta.csv
score "FIG 5 - lambda sweep (validation)"     sweep_lam   fig5_lambda.csv

echo
echo "all results under $OUT/"
