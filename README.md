# Reducing Language-Prior Drift in EMG Silent Speech Recognition via NeuroSymbolic Constrained Decoding with LLMs

Reference implementation for the TASLP manuscript. The manuscript sources and
the compiled PDF are kept outside this repository; everything here is the code
and the scored results behind its tables and figures.

A compact **EMG adapter** is trained on silent facial EMG while a **decoder-only
LLM stays frozen**. Dual supervision (autoregressive + auxiliary CTC) gives two
decoding channels over one adapter representation:

* the **language channel** — adapter → frozen LLM → AR character head;
* the **alignment channel** — adapter → CTC head, which never touches the LLM.

The paper's result is about *where* a symbolic constraint belongs. On the test
split the AR beam is already admitted by the task grammar on **100%** of
utterances, so lexicon and validity constraints applied there are inert: they
leave the output byte-identical. The CTC channel violates the grammar on **40%**
of utterances. Enforcing an utterance-level grammar **inside CTC prefix search**
takes WER from 0.295 to **0.235** at no measurable decoding cost.

The reported system is therefore `--decoder ctc_grammar`. It does not query the
LLM at inference, but it still needs the LLM at *training* time: dropping the AR
objective costs the same decoder 0.235 → 0.311.

---

## Results directories

| path | what it is |
|---|---|
| `Results_Current/` | **authoritative, and the tree shipped here.** The manuscript's tables and figures are scored from these `*.jsonl` outputs and `*.csv` summaries. |
| `Results_reproduced/` | not shipped. `run_experiments.py` writes here by default, so a re-run lands beside the reference tree instead of overwriting it. |

---

## Layout

```
train_emg_llm.py             training (Sec. IV-D, Table I)
inference_emg_llm.py         all ten decoding conditions (Sec. V-B)
split_data.py                400/50/50 split, with manifest pinning

scripts/
  extract_features.py        raw EMG -> 112-dim features (Sec. III-B)
  make_lexicon.py            flat word lexicon from TRAIN transcripts
  grammar.py                 utterance grammar: induce / enumerate / verify (Alg. 1)
  data_utils.py              transcript normalisation + feature z-scoring
  dataset_emg.py             dataset / collation
  augment.py                 train-time EMG augmentation
  ar_decode.py               prefix-KV-cached AR step, scoring, greedy, beam
  ns_utils.py                tries, char 5-gram, CTC forward, AR-side NS beam,
                             and ctc_grammar_beam (Alg. 2)
  tune_ns.py                 validation search over the decoding weights
  metrics.py                 corpus WER/CER/exact-match, decomposition, bootstrap
  evaluate.py                turns *.jsonl outputs into the paper's tables
  analyze_pool.py            pool oracle + channel complementarity
  run_experiments.py         drives every experiment suite
  fig_style.py               shared drawing primitives (auto-fitting text)
  fig1_layout.py             Fig. 1 geometry and labels, in slide inches
  pptx_text.py               mathtext -> Unicode runs, for the deck
  make_fig1.py               fig1_layout -> Overview_Pipeline_Adapter.png
  make_overview_pptx.py      fig1_layout -> Overview.pptx, slide 2
  make_figures.py            Fig. 2 from the scored CSVs (see Figures below)
  make_fig_representation.py Fig. 4 from fig_representation.csv

models/emg_adapter.py        EMGAdapterV2 (Sec. IV-B2)
slurm/                       SLURM jobs; run_all.sh submits the whole graph
```

---

## Pipeline

```bash
pip install -r requirements.txt

# 1) Split 500 silent utterances into 400/50/50 and record the exact partition
python split_data.py --clean --write_manifest artifacts/split_manifest.json

# 2) Symbolic resources, from TRAIN transcripts only
python scripts/make_lexicon.py --src data/train_emg --out artifacts/lexicon.txt

# 3) Train. The primary backbone is Llama-3.2-1B; Table I settings are
#    800 epochs, batch 8, lr 1e-4, wd 0.01, lambda_ctc 0.2, early stop 200.
#    NOTE: train_emg_llm.py's argparse default is --epochs 400. The reported
#    runs were launched from slurm/train.sbatch, which passes 800.
python train_emg_llm.py \
  --base_model meta-llama/Llama-3.2-1B \
  --epochs 800 --early_stop 200 --tag llama32_1b

# 4) Decode the held-out test split with the reported system
python inference_emg_llm.py \
  --ckpt artifacts/best_checkpoint_llama32_1b.pt \
  --data_dir data/test_emg \
  --decoder ctc_grammar \
  --normalizer artifacts/emg_norm.pkl \
  --lexicon artifacts/lexicon.txt \
  --train_dir_for_lm data/train_emg \
  --beta 0.0 --max_len 64 --ctc_grammar_beam 32 --ctc_grammar_pool 8 \
  --out Results_reproduced/decoders/ctc_grammar.jsonl

# 5) Score, and regenerate the figures
python scripts/evaluate.py Results_reproduced/decoders/*.jsonl --bootstrap
python scripts/make_fig1.py                                  # Fig. 1
python scripts/make_figures.py --only fig2                   # Fig. 2
python scripts/make_fig_representation.py                    # Fig. 4
```

To reproduce a split exactly later, pass the recorded manifest:

```bash
python split_data.py --manifest artifacts/split_manifest.json --clean
```

The whole graph (train → tune → decode → score) can be submitted at once:

```bash
bash slurm/run_all.sh          # PRIMARY_TAG defaults to llama32_1b
```

---

## Figures

All five are 300-600 dpi PNG, flattened to RGB, written into the manuscript's
`Figures/` directory. The plots are drawn at the printed IEEE column width
(3.45 in single, 7.16 in double) so `\includegraphics` never rescales them, and
set in STIX, which is metrically Times and carries the math alphabet the labels
use.

| figure | file | built by | source |
|---|---|---|---|
| 1 | `Overview_Pipeline_Adapter.png` | `make_fig1.py` | `scripts/fig1_layout.py` |
| 2 | `Constraint_Placement.png` | `make_figures.py --only fig2` | `table2_decoders.csv`, `fig3_validity.csv`, `table5_decomp.csv` |
| 3 | `LLM_Backbone_Bars.png` | — | `fig2_backbones.csv` plus per-seed and bits-per-character values |
| 4 | `Target_Representation.png` | `make_fig_representation.py` | `fig_representation.csv` |
| 5 | `Sweeps.png` | — | `fig4_beta.csv`, `fig5_lambda.csv`, `fig4c_lambda_ar.csv` |

`make_figures.py --only fig2` reproduces Fig. 2 pixel-for-pixel from
`Results_Current/`. Its Fig. 3 and Fig. 5 code predates the shipped versions of
those two figures, which add the per-seed dots, the bits-per-character panel and
the validation-against-test panel; rebuilding them needs per-seed WERs and
bits-per-character values that are not in this tree, so the committed PNGs are
the reference for both.

Fig. 4 stores its three training seeds in `fig_representation.csv` as integer
word-error counts `k` over the 132 reference words of the test split, so each
WER is exactly `k/132` and the plotted mean and SD agree with Table III to the
last printed digit.

Fig. 1 has an editable PowerPoint copy on **slide 2** of `Figures/Overview.pptx`
(slide 1 is untouched; the deck as submitted is kept beside it as
`Overview_submitted.pptx`):

```bash
python scripts/make_overview_pptx.py    # rewrite slide 2 as native shapes
```

Both the PNG and the slide are generated from `scripts/fig1_layout.py`, so they
cannot drift apart. Edit the layout module and rerun both scripts; editing the
deck by hand is fine for a one-off export, but those edits are overwritten the
next time `make_overview_pptx.py` runs.

---

## Decoding conditions (`--decoder`)

Grouped by the channel that produces the transcript (Sec. V-B, Table II).

| value | condition | test WER |
|---|---|---|
| `ar_greedy` | greedy AR decoding (K = 1) | 0.356 |
| `ar_beam` | AR beam search with length normalisation | 0.333 |
| `ns` | AR beam + trie + 5-gram + boundary + EOS gating | 0.333 |
| `ns_ctc_select` | `ns` + CTC candidate selection (Eq. 18) | 0.333 |
| `ns_joint_rerank` | `ns` + joint AR/CTC reranking at fixed `lambda_fix` | 0.288 |
| `ns_joint_adaptive` | `ns` + joint reranking + adaptive fusion (Eq. 19) | 0.288 |
| `ctc_greedy` | greedy collapse from the CTC head | 0.295 |
| `ctc_lexicon` | CTC prefix beam under the flat word lexicon | 0.235 |
| **`ctc_grammar`** | **CTC prefix beam under the utterance grammar (Alg. 2)** | **0.235** |
| `ctc_grammar_ar` | `ctc_grammar` + AR rescoring of the valid pool | 0.265 |

The first six run on the language channel; `ns` and `ns_ctc_select` are
byte-identical to `ar_beam` on all 50 test utterances, which is the paper's
headline null result and not a bug. `ctc_grammar_ar` wins on validation and
loses on test; it is reported as a negative result.

Component switches (all `BooleanOptionalAction`, so `--no-X` disables):

```bash
--no-trie              # drop the trie constraint
--no-grammar           # use the flat word lexicon instead of the utterance grammar
--no-eos_gating        # drop EOS gating
--beta 0.0             # drop char 5-gram fusion
--kappa 0 --gamma 0    # drop the boundary terms
```

Training-supervision variants (Table III) are set at training time with
`--supervision {ar_ctc,ar_only,ctc_only}`.

---

## Experiment suites

```bash
python scripts/run_experiments.py --suite decoders     # Table II
python scripts/run_experiments.py --suite supervision  # Table III
python scripts/run_experiments.py --suite decomp       # Table IV
python scripts/run_experiments.py --suite latency      # Table V
python scripts/run_experiments.py --suite sweep_beta   # Fig. 5(a), validation
python scripts/run_experiments.py --suite sweep_lam    # Fig. 5(b), validation

# Fig. 3: one checkpoint per frozen backbone, same adapter + decoding recipe
python scripts/run_experiments.py --suite backbones \
  --backbone_ckpt llama32_1b=artifacts/best_checkpoint_llama32_1b.pt \
  --backbone_ckpt llama32_3b=artifacts/best_checkpoint_llama32_3b.pt
```

Add `--dry_run` to print the commands without executing them. The Phase-2
SLURM jobs (`slurm/decode_phase2.sbatch`, `latency_phase2.sbatch`,
`backbones_phase2.sbatch`, `probe_pool.sbatch`) are what produced the
manuscript's `Results_Current/` tree.

Table V requires one job per decoder under identical flags; timings measured
across jobs are not comparable.

---

## Symbolic resources

Everything symbolic is induced from the **400 training transcripts only**.

* **Flat word lexicon** — 910 entries, 889 of them numeral literals. Weak: a
  word trie resets at every space, so it accepts `1045 am am`.
* **Utterance grammar** (`scripts/grammar.py`) — a trie over 54,510 *complete*
  transcripts from three induced templates (`MON D2 D4`, `WD MON D2`,
  `D4 MER`). Slots are filled from **semantic ranges**, never observed literals:
  years span the train interval 1883–2020, days respect month length. Training
  contains only 89 distinct years, and four validation years are absent from it,
  so enumerating observed literals would make those references unreachable.
  `verify_coverage` re-checks this and raises rather than failing quietly.

Validation is used only to confirm the grammar covers it, never to widen it.
The test split is never inspected for either purpose.

---

## Metric convention

WER and CER are **corpus level**: total edit distance over the corpus divided by
the total number of reference words (resp. characters). Sub/Del/Ins are
normalised the same way, so they sum exactly to the total WER. Exact-match is the
percentage of utterances that match after transcript normalisation.

This is the convention the paper's tables use. A per-utterance (macro) average
is a different quantity and gives visibly different numbers on a 50-utterance
set; `scripts/metrics.py` exposes it separately as `macro_wer` if you want it.

Confidence intervals are 95% percentile bootstrap over utterances (10,000
resamples). System comparisons use a **paired** bootstrap over the same budget,
which is the right test because every decoder sees the same 50 utterances
(`evaluate.py --bootstrap` and `--compare`).

An **oracle** row selects, per utterance, the candidate with the lowest WER
against that utterance's reference. It is not a system and cannot be run without
the references; it bounds what any reranker over the same candidates could
achieve. `scripts/analyze_pool.py` computes it.

---

## Transcript normalisation

`TextTransform.clean()` lowercases, applies `unidecode`, **deletes** every
character outside `[a-z0-9]` and whitespace, then collapses whitespace.

Punctuation is deleted, not replaced with a space. This matters for the
closed-vocabulary benchmark: `"10:45 AM"` must normalise to `"1045 am"` (one
token), matching the released reference transcripts. Replacing punctuation with
a space would yield `"10 45 am"`, splitting every time expression into two words
and inflating both the lexicon and the reported WER.

---

## Notes

- The LLM backbone is frozen throughout. Checkpoints store **only** the trained
  modules (adapter, prompts, character embedding, LM head, CTC head); the
  backbone is reloaded from `--base_model`. Legacy checkpoints that embed the
  full backbone still load.
- Decoding caches the fixed EMG prefix's KV once per utterance and re-runs only
  the short character suffix. This is numerically identical to full
  recomputation and is what makes beam-search validation over 800 epochs
  practical.
- The adapter's self-attention is **bidirectional**, so the system is non-causal
  and decodes complete segmented utterances offline. It is not streaming.
- The EMG adapter auto-adapts to the channel count via `in_dim`, inferred from
  the feature files.
- No audio is used for training, decoding, rescoring or evaluation.
