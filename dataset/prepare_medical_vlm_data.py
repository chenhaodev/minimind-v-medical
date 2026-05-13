#!/usr/bin/env python3
"""
Medical VLM dataset preparation script.

Downloads PMC-VQA from HuggingFace, deduplicates on-the-fly, converts to the
parquet schema consumed by VLMDataset (conversations JSON str + image_bytes binary),
and optionally mixes in a small sample from an existing general sft_i2t.parquet.

Output columns:
  conversations  pa.string()  — JSON list of {role, content} dicts; <image> placeholder in user turn
  image_bytes    pa.binary()  — JPEG-compressed image bytes (single image per row)

Fast-SFT recommended usage (~50K unique medical + 2K general):
    python dataset/prepare_medical_vlm_data.py \
        --output_path ./dataset/medical_vlm_sft.parquet \
        --mix_general_ratio 0.05 \
        --general_parquet ./dataset/sft_i2t.parquet

Full dataset (deduplicated):
    python dataset/prepare_medical_vlm_data.py \
        --output_path ./dataset/medical_vlm_sft_full.parquet \
        --max_pmc_vqa 0 \
        --mix_general_ratio 0.05 \
        --general_parquet ./dataset/sft_i2t.parquet
"""

import argparse
import hashlib
import io
import json
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

MEDICAL_SYSTEM_PROMPT = (
    "You are a medical AI assistant specializing in biomedical image analysis. "
    "Answer questions based on the provided image."
)

_IMG_FIELDS = ("Figure", "image", "fig")
_Q_FIELDS = ("Question", "question")
_A_FIELDS = ("Answer", "answer")
_CHOICE_LETTERS = ("A", "B", "C", "D")


def _open_image_field(img_field) -> "Image.Image | None":
    try:
        if isinstance(img_field, dict) and "bytes" in img_field:
            return Image.open(io.BytesIO(img_field["bytes"]))
        if hasattr(img_field, "save"):
            return img_field
    except Exception:
        pass
    return None


def pil_to_bytes(image: Image.Image, fmt: str = "JPEG", quality: int = 85) -> bytes:
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def build_conversations(question: str, answer: str, system_prompt: "str | None" = MEDICAL_SYSTEM_PROMPT) -> list:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": f"<image>\n{question}"})
    msgs.append({"role": "assistant", "content": answer})
    return msgs


def save_to_parquet(rows: list, output_path: str, batch_size: int = 1000) -> None:
    schema = pa.schema([("conversations", pa.string()), ("image_bytes", pa.binary())])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with pq.ParquetWriter(output_path, schema) as writer:
        for start in range(0, len(rows), batch_size):
            batch = rows[start: start + batch_size]
            writer.write_table(pa.table(
                {"conversations": [r["conversations"] for r in batch],
                 "image_bytes":   [r["image_bytes"]   for r in batch]},
                schema=schema,
            ))
    print(f"Saved {len(rows):,} rows → {output_path}")


def download_pmc_vqa(max_samples: "int | None" = None, seed: int = 42, image_quality: int = 85) -> list:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print("Downloading PMC-VQA from HuggingFace (FreedomIntelligence/PMC-VQA)…")
    ds = load_dataset("FreedomIntelligence/PMC-VQA", split="train", trust_remote_code=True)
    print(f"  {len(ds):,} rows, columns: {ds.column_names}")

    rows: list = []
    seen: set = set()
    skipped = 0
    dupes = 0

    for i, row in enumerate(ds):
        if i % 10000 == 0:
            print(f"  Processing PMC-VQA {i:,}/{len(ds):,}…")

        img_field = next((row.get(f) for f in _IMG_FIELDS if row.get(f) is not None), None)
        img = _open_image_field(img_field) if img_field is not None else None
        if img is None:
            skipped += 1
            continue

        question = next((row.get(f) for f in _Q_FIELDS if row.get(f)), "") or ""
        choices = []
        for letter in _CHOICE_LETTERS:
            val = row.get(f"Choice {letter}") or row.get(f"choice_{letter.lower()}") or ""
            if val:
                choices.append(f"{letter}. {val}")
        if choices:
            question = question.rstrip() + "\n" + "\n".join(choices)

        answer = str(next((row.get(f) for f in _A_FIELDS if row.get(f)), None) or "")
        if not question or not answer:
            skipped += 1
            continue

        # Dedup on normalized (question, answer) inline — avoids buffering full dataset
        key = hashlib.md5(
            (" ".join(question.lower().split()) + "|||" + " ".join(answer.lower().split())).encode()
        ).hexdigest()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)

        rows.append({
            "conversations": json.dumps(build_conversations(question, answer), ensure_ascii=False),
            "image_bytes": pil_to_bytes(img, quality=image_quality),
        })

        if max_samples is not None and len(rows) >= max_samples:
            break

    print(f"  PMC-VQA: {len(rows):,} unique rows ({skipped} invalid, {dupes} duplicates removed)")
    return rows


def load_general_sample(parquet_path: str, n_samples: int, seed: int = 42) -> list:
    try:
        pf = pq.ParquetFile(parquet_path)
    except (FileNotFoundError, OSError):
        print(f"WARNING: general parquet not found at {parquet_path}, skipping mix", file=sys.stderr)
        return []

    total = pf.metadata.num_rows
    n_samples = min(n_samples, total)
    print(f"  General sample: {n_samples:,} of {total:,} rows")

    chosen_set = set(random.Random(seed).sample(range(total), n_samples))
    rows: list = []
    offset = 0

    for batch in pf.iter_batches(batch_size=4096):
        batch_len = len(batch)
        for local_idx in range(batch_len):
            if offset + local_idx in chosen_set:
                img = batch.column("image_bytes")[local_idx].as_py()
                rows.append({
                    "conversations": batch.column("conversations")[local_idx].as_py(),
                    "image_bytes": img[0] if isinstance(img, list) else img,
                })
                if len(rows) >= n_samples:
                    break
        offset += batch_len
        if len(rows) >= n_samples:
            break

    print(f"  Loaded {len(rows):,} general rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare medical VLM training data")
    parser.add_argument("--output_path",       default="./dataset/medical_vlm_sft.parquet")
    parser.add_argument("--max_pmc_vqa",       type=int,   default=50000,
                        help="Max unique rows from PMC-VQA after dedup (default: 50000; set to 0 for all)")
    parser.add_argument("--mix_general_ratio", type=float, default=0.05,
                        help="Fraction 0.0–1.0 of final dataset from general parquet (default: 0.05)")
    parser.add_argument("--general_parquet",   default="./dataset/sft_i2t.parquet")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--image_quality",     type=int,   default=85,
                        help="JPEG quality 1–95 (lower = smaller files)")
    args = parser.parse_args()

    max_pmc = args.max_pmc_vqa if args.max_pmc_vqa > 0 else None
    medical_rows = download_pmc_vqa(max_samples=max_pmc, seed=args.seed, image_quality=args.image_quality)

    all_rows = medical_rows
    if args.mix_general_ratio > 0.0:
        n_general = int(len(medical_rows) * args.mix_general_ratio / (1.0 - args.mix_general_ratio))
        all_rows = medical_rows + load_general_sample(args.general_parquet, n_general, seed=args.seed)

    random.Random(args.seed).shuffle(all_rows)
    print(f"Final dataset: {len(all_rows):,} rows  "
          f"({len(medical_rows):,} medical + {len(all_rows) - len(medical_rows):,} general)")

    save_to_parquet(all_rows, args.output_path)


if __name__ == "__main__":
    main()
```
