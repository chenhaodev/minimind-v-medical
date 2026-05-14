# MiniMind-V Medical

A medical-domain Vision-Language Model (VLM) built on top of [MiniMind-V](https://github.com/jingyaogong/minimind-v).

- **LLM backbone**: [`full_sft_medical_768.pth`](https://github.com/chenhaodev/minimind-medical-models) — a 65M-parameter LLM pretrained and SFT'd on bilingual medical text (HuatuoGPT, CMtMedQA, ChatDoctor, MedRAG/textbooks, etc.)
- **Vision encoder**: frozen SigLIP2 (`siglip2-base-p32-256-ve`, 256 × 256 input)
- **VLM training data**: PMC-VQA + SLAKE + VQA-RAD + PathVQA + optional general visual grounding mix
- **No architecture changes** from upstream — only the backbone weights and training data differ

---

## Architecture

```
Image → SigLIP2 (frozen) → MMVisionProjector (MLP) → MiniMind LLM (medical backbone) → Answer
```

| Component | Details |
|---|---|
| LLM | MiniMind, hidden_size=768, 8 layers, GQA (8q/4kv heads), vocab=6400 |
| Vision encoder | SigLIP2-base-p32-256-ve, always frozen |
| Projector | LayerNorm → Linear → GELU → Linear (image_hidden_size → 768) |
| Image tokens | 64 × `<\|image_pad\|>` per image |
| Chat template | Qwen3-style `<\|im_start\|>` / `<\|im_end\|>` |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download vision encoder

```bash
modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir ./model/siglip2-base-p32-256-ve
```

### 3. Download medical LLM backbone

```bash
mkdir -p out
git lfs install
git clone https://github.com/chenhaodev/minimind-medical-models /tmp/medical-models
cp /tmp/medical-models/full_sft_medical_768.pth ./out/
```

---

## Data Preparation

Downloads medical VQA datasets from HuggingFace, deduplicates on-the-fly, converts to parquet, and optionally mixes in a sample from a general `sft_i2t.parquet`.

PMC-VQA multi-choice answers (stored as letters "A"/"B"/"C"/"D") are automatically resolved to full choice text so all sources produce a consistent open-ended format.

The PMC-VQA loader requires HuggingFace `trust_remote_code=True`; only run against trusted dataset sources.

### Recommended dataset (~63K rows)

```bash
python dataset/prepare_medical_vlm_data.py \
    --output_path ./dataset/medical_vlm_sft.parquet \
    --max_pmc_vqa 50000 \
    --max_slake 5000 \
    --max_vqa_rad 3000 \
    --max_path_vqa 5000 \
    --mix_general_ratio 0.1 \
    --general_parquet ./dataset/sft_i2t.parquet
```

### Flag reference

| Flag | Default | Meaning |
|---|---|---|
| `--max_pmc_vqa` | `50000` | Unique PMC-VQA rows after dedup (0 = all ~227K) |
| `--max_slake` | `5000` | Unique SLAKE rows after dedup (0 = skip source) |
| `--max_vqa_rad` | `3000` | Unique VQA-RAD rows after dedup (0 = skip source) |
| `--max_path_vqa` | `5000` | Unique PathVQA rows after dedup (0 = skip source) |
| `--mix_general_ratio` | `0.1` | Fraction in `[0.0, 1.0)` from general parquet; `0` disables mix |
| `--general_parquet` | `./dataset/sft_i2t.parquet` | Source of general VLM data for mixing |
| `--seed` | `42` | Random seed for general-data sampling and shuffle |
| `--image_quality` | `85` | JPEG compression quality (lower = smaller files) |

Deduplication runs on normalized `(question, answer)` per source **before** applying the cap, so `--max_X N` always yields N unique samples.

### PMC-VQA only (original behaviour)

```bash
python dataset/prepare_medical_vlm_data.py \
    --output_path ./dataset/medical_vlm_sft.parquet \
    --max_slake 0 \
    --max_vqa_rad 0 \
    --max_path_vqa 0 \
    --mix_general_ratio 0.1 \
    --general_parquet ./dataset/sft_i2t.parquet
```

---

## Training

### Option A — Direct SFT (recommended)

Start from the medical LLM backbone and train the vision projector plus the first and last LLM layers (`freeze_llm=1`), keeping the middle layers' medical knowledge frozen.

```bash
python trainer/train_sft_vlm.py \
    --from_weight full_sft_medical \
    --save_weight sft_vlm_medical \
    --data_path dataset/medical_vlm_sft.parquet \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --warmup_steps 200 \
    --freeze_llm 1 \
    --max_seq_len 768 \
    --accumulation_steps 4 \
    --save_dir out
```

Output: `out/sft_vlm_medical_768.pth` (latest) and `out/sft_vlm_medical_best_768.pth` (best val loss)

### Option B — Two-stage (conservative alignment)

Stage 1 gives the projector broad visual grounding on general data without touching any LLM layers. Stage 2 then fine-tunes on medical data.

**Stage 1 — General-V projector alignment** (`freeze_llm=2`, projector only)

```bash
python trainer/train_pretrain_vlm.py \
    --from_weight full_sft_medical \
    --save_weight pretrain_vlm_general \
    --data_path dataset/sft_i2t.parquet \
    --epochs 1 \
    --batch_size 16 \
    --learning_rate 4e-4 \
    --warmup_steps 200 \
    --freeze_llm 2 \
    --max_seq_len 512 \
    --accumulation_steps 2 \
    --save_dir out
```

Output: `out/pretrain_vlm_general_768.pth` — LLM layers untouched.

**Stage 2 — Medical SFT** (`freeze_llm=1`, projector + first/last LLM layers)

```bash
python trainer/train_sft_vlm.py \
    --from_weight pretrain_vlm_general \
    --save_weight sft_vlm_medical \
    --data_path dataset/medical_vlm_sft.parquet \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --warmup_steps 200 \
    --freeze_llm 1 \
    --max_seq_len 768 \
    --accumulation_steps 4 \
    --save_dir out
```

Output: `out/sft_vlm_medical_768.pth` + `out/sft_vlm_medical_best_768.pth`

### Resume after interruption

Add `--from_resume 1` to any training command to continue from the last checkpoint.

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=<N> trainer/train_sft_vlm.py \
    --from_weight full_sft_medical \
    --save_weight sft_vlm_medical \
    --data_path dataset/medical_vlm_sft.parquet \
    --epochs 2 --batch_size 4 --learning_rate 5e-6 \
    --warmup_steps 200 --freeze_llm 1 --max_seq_len 768 \
    --accumulation_steps 4 --save_dir out
```

### Training flag reference

| Flag | Default | Meaning |
|---|---|---|
| `--warmup_steps` | `200` | Linear LR warmup steps before cosine decay |
| `--val_ratio` | `0.05` | Fraction of dataset held out for validation |
| `--validate_interval` | `1` | Run validation every N epochs |
| `--freeze_llm` | — | See table below |

### `freeze_llm` reference

| Value | Trainable parameters |
|---|---|
| `0` | All layers except vision encoder |
| `1` | `vision_proj` + LLM layers 0 and 7 (first/last) |
| `2` | `vision_proj` only |

---

## Inference

```bash
python eval_vlm.py \
    --load_from model \
    --weight sft_vlm_medical \
    --save_dir out \
    --image_dir ./dataset/eval_images/
```

Place medical images (X-rays, pathology slides, biomedical figures) in `./dataset/eval_images/` to test the model.

---

## Dataset Format

All parquet files use two columns:

| Column | Type | Content |
|---|---|---|
| `conversations` | `string` | JSON array of `{role, content}` dicts; user turn contains `<image>` placeholder |
| `image_bytes` | `binary` | JPEG-compressed image bytes |

The `<image>` placeholder is expanded at runtime to 64 consecutive `<|image_pad|>` tokens by `VLMDataset`.

Preview samples from any parquet:

```bash
cd dataset && python lm_dataset.py  # edit path inside __main__ to point at medical_vlm_sft.parquet
```

---

## File Overview

```
minimind-v-medical/
├── dataset/
│   ├── prepare_medical_vlm_data.py   # Download + dedup PMC-VQA / SLAKE / VQA-RAD / PathVQA
│   ├── lm_dataset.py                 # VLMDataset (unchanged from upstream)
│   └── medical_vlm_sft.parquet       # Generated training data (gitignored)
├── model/
│   ├── model_minimind.py             # LLM backbone (MiniMind)
│   ├── model_vlm.py                  # VLM wrapper (MiniMindVLM)
│   └── siglip2-base-p32-256-ve/      # Vision encoder weights (download separately)
├── trainer/
│   ├── train_pretrain_vlm.py         # Stage 1 training (projector alignment)
│   ├── train_sft_vlm.py              # Stage 2 / direct SFT (with validation loop)
│   └── trainer_utils.py              # init_vlm_model, checkpointing, freeze logic
├── tests/
│   └── test_prepare_medical_vlm_data.py  # Unit tests for dataset preparation
├── out/
│   ├── full_sft_medical_768.pth      # Input: medical LLM backbone
│   ├── sft_vlm_medical_768.pth       # Output: latest checkpoint
│   └── sft_vlm_medical_best_768.pth  # Output: best val-loss checkpoint
├── eval_vlm.py                       # CLI inference
└── scripts/
    ├── convert_vlm.py                # PyTorch ↔ Transformers format conversion
    └── web_demo_vlm.py               # Gradio WebUI
```

---

## Credits

- VLM architecture and training pipeline: [MiniMind-V](https://github.com/jingyaogong/minimind-v) by [@jingyaogong](https://github.com/jingyaogong)
- Medical LLM backbone: [minimind-medical](https://github.com/chenhaodev/minimind-medical) / [minimind-medical-models](https://github.com/chenhaodev/minimind-medical-models)
- Training data:
  - [PMC-VQA](https://huggingface.co/datasets/FreedomIntelligence/PMC-VQA) (FreedomIntelligence) — biomedical figures from PubMed (~227K)
  - [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) (BoKelvin) — multi-organ radiology EN+ZH (~14K)
  - [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) (flaviagiammarino) — radiology VQA (~3.5K)
  - [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) (flaviagiammarino) — pathology VQA (~32K)
