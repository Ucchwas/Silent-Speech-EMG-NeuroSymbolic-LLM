# EMG Silent Speech with NeuroSymbolic Decoding and Frozen LLMs

This repository contains the code for the EMG silent-speech recognition pipeline described in **“Reducing Language-Prior Drift in EMG Silent Speech Recognition via NeuroSymbolic Constrained Decoding with Large Language Models.”** The system maps frame-level facial EMG features to a frozen decoder-only LLM through a compact trainable EMG adapter, trains with **dual AR+CTC supervision**, and applies **NeuroSymbolic constrained decoding** at inference using a lexicon trie, character 5-gram fusion, boundary control, EOS gating, and optional CTC-assisted reranking/adaptive fusion. In the paper, the full system on the Silent Speech EMG closed-vocabulary silent benchmark improves WER from **0.273** under AR beam decoding to **0.189**, with **0.073 CER** and **72% exact-match** using the Meta Llama 3.2 3B backbone. The same inference recipe is also evaluated with **Meta Llama 3.2 1B**, **Qwen2.5-3B-Instruct**, and **Mistral-7B-Instruct v0.3**.  

This README is organized to make the repository easy to run end to end: environment setup, data split, training, lexicon generation, inference, and result organization.

---

## 1. What this project does

The repository implements the following pipeline:

1. **EMG preprocessing and framing** into fixed-dimensional frame-level features.
2. **EMGAdapterV2** to map the EMG feature sequence into LLM-sized embeddings.
3. A **frozen decoder-only LLM** conditioned on EMG embeddings for character-level autoregressive decoding.
4. An **auxiliary CTC head** attached to the adapter stream for alignment-sensitive supervision and complementary inference-time scoring.
5. **NeuroSymbolic decoding** built from training transcripts only:
   - closed-vocabulary lexicon
   - prefix trie
   - character 5-gram LM
   - optional CTC-assisted reranking and adaptive AR/CTC fusion

The dataset setup used in the paper is the **Silent Speech EMG closed-vocabulary silent split** with **500 utterances**, partitioned into **400 train / 50 validation / 50 test**. The benchmark uses **8-channel facial EMG** sampled at **1 kHz**. The fixed preprocessing recipe uses a **27 ms window** and **10 ms hop**, and features are z-normalized using statistics fit on the training split only.

---

## 2. Repository layout

```text
Silent_Speech_EMG/
├── artifacts/                # generated training/inference artifacts
├── data/                     # train/val/test EMG features + metadata
├── Figures/                  # paper figures (not needed to run code)
├── models/                   # local model folders and adapter module
├── Papers/                   # manuscript/pdf material (not needed to run code)
├── Results/                  # saved tables/plots/json outputs for experiments
├── scripts/                  # helper scripts (dataset, lexicon, utilities)
├── inference_emg_llm.py      # inference + evaluation
├── split_data.py             # dataset split/cleanup helper
├── train_emg_llm.py          # training script
├── README.md
└── requirements              # Python dependencies
```

---

## 3. Environment setup

Go to the project root and activate your environment.

```bash
cd Silent_Speech_EMG
conda activate ss2
```

Install dependencies:

```bash
pip install -r requirements
```

If your dependency file is named `requirements.txt` locally, use:

```bash
pip install -r requirements.txt
```

---

## 4. Optional cleanup of previous artifacts

If you want a fresh run, remove old generated files first.

### Linux / macOS
```bash
rm -rf artifacts
mkdir -p artifacts
```

### Windows PowerShell
```powershell
if (Test-Path artifacts) { Remove-Item -Recurse -Force artifacts }
New-Item -ItemType Directory -Path artifacts | Out-Null
```

This is optional, but recommended before a new training run.

---

## 5. Data split

Prepare the train/validation/test folders.

```bash
python split_data.py --clean
```

After this step, the code expects data organized like:

```text
data/train_emg/*.npy + *.json
data/val_emg/*.npy + *.json
data/test_emg/*.npy + *.json
```

---

## 6. Training

The default training recipe follows the paper’s neural setup:

- frozen decoder-only LLM backbone
- trainable EMG adapter
- AR + CTC dual supervision
- batch size 8
- 250 epochs
- AdamW, learning rate `1e-4`
- weight decay `0.01`
- gradient clip `1.0`
- `lambda_ctc = 0.2`

A normalizer is fit on the training split only and written to `artifacts/emg_norm.pkl`. Training also writes checkpoints and the saved run configuration.

### Default training command
```bash
python train_emg_llm.py \
  --train_dir data/train_emg \
  --val_dir data/val_emg \
  --artifacts_dir artifacts
```

### Train with the primary paper backbone
```bash
python train_emg_llm.py --base_model meta-llama/Llama-3.2-3B
```

### Train with the other backbones used in the paper
```bash
python train_emg_llm.py --base_model meta-llama/Llama-3.2-1B
python train_emg_llm.py --base_model Qwen/Qwen2.5-3B-Instruct
python train_emg_llm.py --base_model mistralai/Mistral-7B-Instruct-v0.3
```

### Main files written to `artifacts/`

```text
artifacts/emg_norm.pkl
artifacts/latest_checkpoint.pt
artifacts/best_checkpoint.pt
artifacts/train_config.json
```

---

## 7. Build the lexicon for NeuroSymbolic decoding

The NS decoder uses resources derived from the training transcripts.

### Option A
```bash
python scripts/make_lexicon.py --src data/train_emg --out artifacts/lexicon.txt
```

### Option B
If your helper script uses `--train_dir` / `--out_dir`, use:

```bash
python scripts/make_lexicon.py --train_dir data/train_emg --out_dir artifacts
```

Expected output:

```text
artifacts/lexicon.txt
```

If the script also writes training texts for the character LM, keep those in `artifacts/` as well.

---

## 8. Optional cache cleanup for NS resources

If you want to rebuild trie / LM caches from scratch:

### Linux / macOS
```bash
rm -f artifacts/trie.pkl artifacts/char_5gram.pkl artifacts/lexicon_words.pkl
```

### Windows PowerShell
```powershell
Remove-Item -Force artifacts/trie.pkl, artifacts/char_5gram.pkl, artifacts/lexicon_words.pkl -ErrorAction SilentlyContinue
```

---

## 9. Inference

The inference script always computes:

- AR greedy
- AR beam
- CTC greedy

If `--ns` is enabled, it also runs NeuroSymbolic decoding. If `--rerank --joint --adaptive` are enabled, it performs CTC-assisted reranking / adaptive fusion over the candidate set implemented by the current script.

> **Important note:** the paper reports some inference-time hyperparameters such as explicit `δ`, `κ`, `γ`, EOS gating `m`, and rerank-list tuning. The current uploaded `inference_emg_llm.py` exposes the following CLI flags directly: `--max_len`, `--min_len`, `--beam`, `--alpha`, `--ns`, `--beta`, `--lexicon`, `--train_texts`, `--rerank`, `--joint`, `--lambda_ctc`, and `--adaptive`. The commands below therefore document the **current repository script interface**.

### A. Validation inference: neural-only AR beam

This is the simplest validation-time decoding run.

```bash
python inference_emg_llm.py \
  --ckpt artifacts/best_checkpoint.pt \
  --data_dir data/val_emg \
  --normalizer artifacts/emg_norm.pkl \
  --out artifacts/infer_val_ar_beam.jsonl \
  --beam 6 \
  --alpha 0.6 \
  --max_len 128
```

You can also run with defaults:

```bash
python inference_emg_llm.py
```

### B. Validation inference: NS constrained beam

Make sure the lexicon exists first.

```bash
python scripts/make_lexicon.py --src data/train_emg --out artifacts/lexicon.txt
```

Then run NS decoding:

```bash
python inference_emg_llm.py \
  --ckpt artifacts/best_checkpoint.pt \
  --data_dir data/val_emg \
  --normalizer artifacts/emg_norm.pkl \
  --out artifacts/infer_val_ns.jsonl \
  --ns \
  --lexicon artifacts/lexicon.txt \
  --train_texts artifacts/train_texts.txt \
  --beam 6 \
  --alpha 0.6 \
  --beta 0.55 \
  --max_len 64 \
  --min_len 8
```

### C. Validation inference: NS + joint reranking + adaptive fusion

```bash
python inference_emg_llm.py \
  --ckpt artifacts/best_checkpoint.pt \
  --data_dir data/val_emg \
  --normalizer artifacts/emg_norm.pkl \
  --out artifacts/infer_val_ns_rerank_adaptive.jsonl \
  --ns \
  --lexicon artifacts/lexicon.txt \
  --train_texts artifacts/train_texts.txt \
  --beam 6 \
  --alpha 0.6 \
  --beta 0.55 \
  --max_len 64 \
  --min_len 8 \
  --rerank \
  --joint \
  --adaptive \
  --lambda_ctc 0.25
```

### Output files written by inference

Typical outputs are JSONL files such as:

```text
artifacts/infer_val_ar_beam.jsonl
artifacts/infer_val_ns.jsonl
artifacts/infer_val_ns_rerank_adaptive.jsonl
```

Each record stores the utterance ID, reference, final hypothesis, and the intermediate decoding outputs written by the script.

---

## 10. Reproducibility notes

This repository is intended to support both training and inference-based reproducibility.

### Validation / checkpoint-selection settings used in training

The training script selects `best_checkpoint.pt` using **neural-only AR beam** validation decoding, consistent with the paper’s checkpoint-selection strategy.

### Paper-reported primary settings

For the best-performing Meta Llama 3.2 3B system, the paper reports:

- validation checkpoint selection under AR beam: `K = 6`, `Lmax = 128`, `alpha = 0.6`, `delta = 5.0`
- test-time NS decoding: trie on, `beta = 0.55`, `kappa = 0.40`, `gamma = 0.45`, `alpha = 0.60`, `delta = 5.0`, `m = 8`, `K = 6`, `Lmax = 64`
- CTC-assisted inference tuned on validation and then frozen for test evaluation

If you are reproducing the paper exactly, keep those settings in mind when aligning the final decoding configuration.

---

## 11. Results summary

Below is a concise summary of the main results reported in the paper.

### 11.1 Primary backbone: Meta Llama 3.2 3B, one fixed checkpoint

| Decoding condition | WER | CER | Exact-match (%) |
|---|---:|---:|---:|
| CTC greedy | 0.326 | 0.109 | 44 |
| AR greedy (K = 1) | 0.288 | 0.105 | 50 |
| AR beam (K = 6) | 0.273 | 0.099 | 54 |
| NS constrained beam | 0.220 | 0.082 | 66 |
| NS + CTC candidate selection | 0.205 | 0.076 | 68 |
| NS + joint AR+CTC reranking | 0.197 | 0.074 | 70 |
| NS + joint reranking + adaptive fusion | 0.189 | 0.073 | 72 |

### 11.2 Backbone comparison under the same inference recipe

| Backbone | WER | CER | Exact-match (%) |
|---|---:|---:|---:|
| Meta Llama 3.2 1B | 0.205 | 0.074 | 56 |
| Meta Llama 3.2 3B | 0.189 | 0.073 | 72 |
| Qwen2.5-3B-Instruct | 0.182 | 0.069 | 58 |
| Mistral-7B-Instruct v0.3 | 0.174 | 0.067 | 60 |

### 11.3 Training-supervision ablation

| Training supervision | WER | CER |
|---|---:|---:|
| CTC-only (no AR head) | 0.338 | 0.113 |
| AR-only (`lambda_ctc = 0`) | 0.206 | 0.077 |
| AR+CTC (default) | 0.189 | 0.073 |

### 11.4 NeuroSymbolic decoding ablation

| Decoder variant | WER | CER | Exact-match (%) |
|---|---:|---:|---:|
| NS constrained beam (full) | 0.220 | 0.082 | 66 |
| w/o trie constraint | 0.250 | 0.078 | 46 |
| w/o 5-gram fusion (`beta = 0`) | 0.227 | 0.113 | 50 |
| w/o boundary terms (`kappa = gamma = 0`) | 0.235 | 0.074 | 52 |
| w/o EOS gating | 0.227 | 0.118 | 52 |

### 11.5 Latency note

The abstract reports a modest decoding-latency increase from **38 ms** to **49 ms** per utterance when moving from AR beam to the full system.

---

## 12. Results folder

I could not inspect the actual local `Results/` folder contents from this environment, so the list below documents the result groups that should be stored there for a clean release of the project.

Recommended organization:

```text
Results/
├── decoding_pipeline/            # Table II style summaries
├── backbone_comparison/          # Fig. 2 style summaries/plots
├── training_ablation/            # Table III style summaries
├── ns_ablation/                  # Table IV + bootstrap summaries
├── latency_memory/               # decoding cost / memory summaries
└── json_outputs/                 # copied or linked inference .jsonl files
```

Good files to keep in `Results/` include:

- summary tables in `.csv` or `.xlsx`
- plots corresponding to backbone comparison and NS ablation
- decoded JSONL outputs copied from `artifacts/`
- any WER/CER/exact-match summaries used in the paper

---

## 13. Notes

- `Figures/` and `Papers/` are not needed to run the code and can be excluded from Git if desired.
- Large local model folders under `models/` should also not be committed to GitHub.
- The repository is most reproducible when you provide:
  - exact model ID or local model path
  - fixed environment versions
  - the exact inference command used for each result table
  - saved checkpoints when redistribution is allowed

---

## 14. Citation

If you use this repository, please cite the associated paper.
