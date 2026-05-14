# MiniMind-V Medical

A medical-domain Vision-Language Model (VLM) built on top of [MiniMind-V](https://github.com/jingyaogong/minimind-v).

- **LLM backbone**: [`full_sft_medical_768.pth`](https://github.com/chenhaodev/minimind-medical-models) — a 65M-parameter LLM pretrained and SFT'd on bilingual medical text (HuatuoGPT, CMtMedQA, ChatDoctor, MedRAG/textbooks, etc.)
- **Vision encoder**: frozen SigLIP2 (`siglip2-base-p32-256-ve`, 256 × 256 input)
- **VLM training data**: PMC-VQA (biomedical figures from PubMed) + small general visual grounding mix
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

Download and convert [PMC-VQA](https://huggingface.co/datasets/FreedomIntelligence/PMC-VQA) to the training parquet format, with deduplication and an optional general visual grounding mix.
The script enables HuggingFace `trust_remote_code=True` for the PMC-VQA
loader, so only run it against trusted dataset sources.

### Fast-SFT dataset (~52K rows, recommended)

```bash
python dataset/prepare_medical_vlm_data.py \
    --output_path ./dataset/medical_vlm_sft.parquet \
    --max_pmc_vqa 50000 \
    --mix_general_ratio 0.1 \
    --general_parquet ./dataset/sft_i2t.parquet
```

| Flag | Default | Meaning |
|---|---|---|
| `--max_pmc_vqa` | `50000` | Unique PMC-VQA rows after dedup (0 = all ~227K) |
| `--mix_general_ratio` | `0.1` | Fraction in `[0.0, 1.0)` from general parquet; `0` disables mix |
| `--general_parquet` | `./dataset/sft_i2t.parquet` | Source of general VLM data for mixing |
| `--image_quality` | `85` | JPEG compression quality (lower = smaller files) |

The script deduplicates on normalized `(question, answer)` **before** applying the cap, so `--max_pmc_vqa N` always yields N unique samples.

### Full deduplicated dataset

```bash
python dataset/prepare_medical_vlm_data.py \
    --output_path ./dataset/medical_vlm_sft_full.parquet \
    --max_pmc_vqa 0 \
    --mix_general_ratio 0.1 \
    --general_parquet ./dataset/sft_i2t.parquet
```

---

## Training

Per the upstream README recommendation, go **directly to SFT** from the medical LLM backbone (`freeze_llm=1` trains the vision projector plus the first and last LLM layers, preserving the middle layers' medical knowledge).

### Option A — Direct SFT (recommended)

```bash
python trainer/train_sft_vlm.py \
    --from_weight full_sft_medical \
    --save_weight sft_vlm_medical \
    --data_path dataset/medical_vlm_sft.parquet \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --freeze_llm 1 \
    --max_seq_len 768 \
    --accumulation_steps 4 \
    --save_dir out
```

Output: `out/sft_vlm_medical_768.pth`

### Option B — Two-stage (conservative alignment)

Use this if you want the vision projector explicitly aligned to the medical embedding space before unfreezing any LLM layers.

**Stage 1 — Projector alignment** (`freeze_llm=2`, projector only)

```bash
python trainer/train_pretrain_vlm.py \
    --from_weight full_sft_medical \
    --save_weight pretrain_vlm_medical \
    --data_path dataset/medical_vlm_sft.parquet \
    --epochs 1 \
    --batch_size 16 \
    --learning_rate 4e-4 \
    --freeze_llm 2 \
    --max_seq_len 512 \
    --accumulation_steps 2 \
    --save_dir out
```

**Stage 2 — SFT** (`freeze_llm=1`, projector + first/last LLM layers)

```bash
python trainer/train_sft_vlm.py \
    --from_weight pretrain_vlm_medical \
    --save_weight sft_vlm_medical \
    --data_path dataset/medical_vlm_sft.parquet \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --freeze_llm 1 \
    --max_seq_len 768 \
    --accumulation_steps 4 \
    --save_dir out
```

### Resume after interruption

Add `--from_resume 1` to any training command to continue from the last checkpoint.

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=<N> trainer/train_sft_vlm.py \
    --from_weight full_sft_medical \
    --save_weight sft_vlm_medical \
    --data_path dataset/medical_vlm_sft.parquet \
    --epochs 2 --batch_size 4 --learning_rate 5e-6 \
    --freeze_llm 1 --max_seq_len 768 --accumulation_steps 4 --save_dir out
```

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
│   ├── prepare_medical_vlm_data.py   # Download + dedup + convert PMC-VQA
│   ├── lm_dataset.py                 # VLMDataset (unchanged from upstream)
│   └── medical_vlm_sft.parquet       # Generated training data (gitignored)
├── model/
│   ├── model_minimind.py             # LLM backbone (MiniMind)
│   ├── model_vlm.py                  # VLM wrapper (MiniMindVLM)
│   └── siglip2-base-p32-256-ve/      # Vision encoder weights (download separately)
├── trainer/
│   ├── train_pretrain_vlm.py         # Stage 1 training (projector alignment)
│   ├── train_sft_vlm.py              # Stage 2 / direct SFT
│   └── trainer_utils.py              # init_vlm_model, checkpointing, freeze logic
├── out/
│   ├── full_sft_medical_768.pth      # Input: medical LLM backbone
│   └── sft_vlm_medical_768.pth       # Output: trained medical VLM
├── eval_vlm.py                       # CLI inference
└── scripts/
    ├── convert_vlm.py                # PyTorch ↔ Transformers format conversion
    └── web_demo_vlm.py               # Gradio WebUI
```

---

## Credits

- VLM architecture and training pipeline: [MiniMind-V](https://github.com/jingyaogong/minimind-v) by [@jingyaogong](https://github.com/jingyaogong)
- Medical LLM backbone: [minimind-medical](https://github.com/chenhaodev/minimind-medical) / [minimind-medical-models](https://github.com/chenhaodev/minimind-medical-models)
- Training data: [PMC-VQA](https://huggingface.co/datasets/FreedomIntelligence/PMC-VQA) (FreedomIntelligence)
