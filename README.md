# EMG Silent Speech with NeuroSymbolic Decoding and Frozen LLMs

This repository contains the code for the EMG silent-speech recognition pipeline described in **“Reducing Language-Prior Drift in EMG Silent Speech Recognition via NeuroSymbolic Constrained Decoding with Large Language Models.”** The system maps frame-level facial EMG features to a frozen decoder-only LLM through a compact trainable EMG adapter, trains with **dual AR+CTC supervision**, and applies **NeuroSymbolic constrained decoding** at inference using a lexicon trie, character 5-gram fusion, boundary control, EOS gating, and optional CTC-assisted reranking/adaptive fusion.

---

## 1. Project Repos

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

We used publicly available **SilentSpeech EMG** dataset with **500 utterances**, partitioned into **400 train / 50 validation / 50 test**. The benchmark uses **8-channel facial EMG** sampled at **1 kHz**.

---

## 2. Repository layout

```text
Silent_Speech_EMG/
├── artifacts/                # generated training/inference artifacts
├── data/                     # train/val/test EMG features + metadata
├── Figures/                  # Figures
├── models/                   # local model folders and adapter module
├── Papers/                   # manuscript/pdf material
├── Results/                  # saved json outputs for experiments
├── scripts/                  # helper scripts (dataset, lexicon, utilities)
├── inference_emg_llm.py      # inference + evaluation
├── split_data.py             # dataset split
├── train_emg_llm.py          # training script
├── README.md
└── requirements              # Python dependencies
```

---

## 3. Environment setup

Go to the project root and activate environment.

```bash
cd Silent_Speech_EMG
conda activate ss2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Optional cleanup of previous artifacts

For a fresh run, remove old generated files first.

### Linux / macOS
```bash
rm -rf artifacts
mkdir -p artifacts
```

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

The default training recipe:

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

### Train with the primary backbone
```bash
python train_emg_llm.py --base_model meta-llama/Llama-3.2-3B
```

### Train with the other backbones
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

```bash
python scripts/make_lexicon.py --src data/train_emg --out artifacts/lexicon.txt
```

If the script also writes training texts for the character LM, keep those in `artifacts/` as well.

---

## 8. Optional cache cleanup for NS resources
To rebuild trie / LM caches from scratch:

### Linux / macOS
```bash
rm -f artifacts/trie.pkl artifacts/char_5gram.pkl artifacts/lexicon_words.pkl
```

---

## 9. Inference

The inference script always computes:

- AR greedy
- AR beam
- CTC greedy

If `--ns` is enabled, it also runs NeuroSymbolic decoding. If `--rerank --joint --adaptive` are enabled, it performs CTC-assisted reranking / adaptive fusion over the candidate set implemented by the current script.


### A. Validation inference: neural-only AR beam

Default run:

```bash
python inference_emg_llm.py
```

or simplest validation-time decoding run with args:

```bash
python inference_emg_llm.py \
  --ckpt artifacts/best_checkpoint.pt \
  --data_dir data/val_emg \
  --normalizer artifacts/emg_norm.pkl \
  --out artifacts/infer_val_ar_beam.jsonl \
  --beam 6 --lmax 128 --alpha 0.6 --delta 5.0
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
  --ns --lexicon artifacts/lexicon.txt \
  --trie --beta 0.55 --kappa 0.40 --gamma 0.45 \
  --beam 6 --lmax 64 --alpha 0.60 --delta 5.0 --min_eos_len 8
```

### C. Validation inference: NS + joint reranking + adaptive fusion

```bash
python inference_emg_llm.py \
  --ckpt artifacts/best_checkpoint.pt \
  --data_dir data/val_emg \
  --normalizer artifacts/emg_norm.pkl \
  --out artifacts/infer_val_ns_rerank_adaptive.jsonl \
  --ns --lexicon artifacts/lexicon.txt \
  --trie --beta 0.55 --kappa 0.40 --gamma 0.45 \
  --beam 6 --lmax 64 --alpha 0.60 --delta 5.0 --min_eos_len 8 \
  --rerank --joint --adaptive \
  --rerank_M 16 --lambda_fix 0.25 \
  --lambda_min 0.0 --lambda_max 0.6 --lambda0 0.25 --a 0.20 --b 0.20
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

The training script selects `best_checkpoint.pt` using **neural-only AR beam** validation decoding.

### Primary settings

For the best-performing Meta Llama 3.2-3B system:

- validation checkpoint selection under AR beam: `K = 6`, `Lmax = 128`, `alpha = 0.6`, `delta = 5.0`
- test-time NS decoding: trie on, `beta = 0.55`, `kappa = 0.40`, `gamma = 0.45`, `alpha = 0.60`, `delta = 5.0`, `m = 8`, `K = 6`, `Lmax = 64`
- CTC-assisted inference tuned on validation and then frozen for test evaluation

---

## 11. Results folder

The `Results/` folder contains the saved experiment outputs. Please visit this folder directly to see the result files and JSON outputs.

Current result groups include folders such as:

- `Ablation_NS_Decoding/`
- `LLM_Backbone_Results/`
- `Silent Speech Test_Decoders/`
- `WER decomposition/`

Example files inside these folders include JSON result files such as:

- `ablation_ns_full.json`
- `ablation_wo_trie.json`
- `llama32_3b.json`
- `qwen25_3b_instruct.json`
- `mistral7b_v03.json`

These files store decoded outputs and experiment summaries for backbone comparison, NeuroSymbolic decoding ablations, decoder evaluation, and WER-related analysis.

