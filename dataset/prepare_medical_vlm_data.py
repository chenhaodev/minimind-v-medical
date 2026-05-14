#!/usr/bin/env python3
"""
Medical VLM dataset preparation script.

Downloads medical VQA datasets from HuggingFace, deduplicates on-the-fly, converts to the
parquet schema consumed by VLMDataset (conversations JSON str + image_bytes binary),
and optionally mixes in a small sample from an existing general sft_i2t.parquet.

Output columns:
  conversations  pa.string()  — JSON list of {role, content} dicts; <image> placeholder in user turn
  image_bytes    pa.binary()  — JPEG-compressed image bytes (single image per row)

Sources:
  PMC-VQA  (FreedomIntelligence/PMC-VQA)   — biomedical figures from PubMed (~227K)
  SLAKE    (BoKelvin/SLAKE)                 — multi-organ radiology EN+ZH (~14K)
  VQA-RAD  (flaviagiammarino/vqa-rad)       — radiology VQA (~3.5K)
  PathVQA  (flaviagiammarino/path-vqa)      — pathology VQA (~32K)

Fast-SFT recommended usage (~63K medical + ~7K general):
  Defaults: PMC-VQA 50K + SLAKE 5K + VQA-RAD 3K + PathVQA 5K, mix_general_ratio=0.1
    python dataset/prepare_medical_vlm_data.py \
        --output_path ./dataset/medical_vlm_sft.parquet \
        --mix_general_ratio 0.1 \
        --general_parquet ./dataset/sft_i2t.parquet

PMC-VQA only:
    python dataset/prepare_medical_vlm_data.py \
        --output_path ./dataset/medical_vlm_sft.parquet \
        --max_slake 0 --max_vqa_rad 0 --max_path_vqa 0 \
        --mix_general_ratio 0.1 \
        --general_parquet ./dataset/sft_i2t.parquet

Full deduplicated dataset (all sources):
    python dataset/prepare_medical_vlm_data.py \
        --output_path ./dataset/medical_vlm_sft_full.parquet \
        --max_pmc_vqa 0 --max_slake 0 --max_vqa_rad 0 --max_path_vqa 0 \
        --mix_general_ratio 0.1 \
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
_REQUIRED_GENERAL_COLUMNS = ("conversations", "image_bytes")

_SLAKE_IMG_FIELDS = ("img_content", "image", "img")
_HF_VQA_IMG_FIELDS = ("image", "img")


def _parse_mix_general_ratio(value: str) -> float:
    """Parse --mix_general_ratio; must be [0.0, 1.0) to avoid div-by-zero."""
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "mix_general_ratio must be a float"
        ) from exc
    if ratio < 0.0 or ratio >= 1.0:
        raise argparse.ArgumentTypeError(
            "mix_general_ratio must satisfy 0.0 <= r < 1.0 "
            "(r=0 disables general mix; r=1 is invalid)"
        )
    return ratio


def _parse_image_quality(value: str) -> int:
    """Parse Pillow JPEG quality; valid values are 1 through 95."""
    try:
        quality = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "image_quality must be an integer"
        ) from exc
    if quality < 1 or quality > 95:
        raise argparse.ArgumentTypeError(
            "image_quality must satisfy 1 <= quality <= 95"
        )
    return quality


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

    print("Downloading PMC-VQA from HuggingFace (xmcmic/PMC-VQA)…")
    ds = load_dataset("xmcmic/PMC-VQA", split="train")
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
        choice_texts: list = []
        for letter in _CHOICE_LETTERS:
            val = row.get(f"Choice {letter}") or row.get(f"choice_{letter.lower()}") or ""
            if val:
                choice_texts.append(val)

        answer = str(next((row.get(f) for f in _A_FIELDS if row.get(f)), None) or "")
        answer_letter = answer.strip().upper()
        if answer_letter in _CHOICE_LETTERS and choice_texts:
            idx = _CHOICE_LETTERS.index(answer_letter)
            if idx < len(choice_texts):
                answer = choice_texts[idx]
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


def _download_hf_vqa(
    hf_id: str,
    name: str,
    img_fields: tuple,
    max_samples: "int | None" = None,
    image_quality: int = 85,
    log_interval: int = 1000,
) -> list:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {name} from HuggingFace ({hf_id})…")
    ds = load_dataset(hf_id, split="train", trust_remote_code=True)
    print(f"  {len(ds):,} rows, columns: {ds.column_names}")

    rows: list = []
    seen: set = set()
    skipped = 0
    dupes = 0

    for i, row in enumerate(ds):
        if i % log_interval == 0:
            print(f"  Processing {name} {i:,}/{len(ds):,}…")

        img_field = next((row.get(f) for f in img_fields if row.get(f) is not None), None)
        img = _open_image_field(img_field) if img_field is not None else None
        if img is None:
            skipped += 1
            continue

        question = next((row.get(f) for f in _Q_FIELDS if row.get(f)), "") or ""
        answer = str(next((row.get(f) for f in _A_FIELDS if row.get(f)), None) or "")
        if not question or not answer:
            skipped += 1
            continue

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

    print(f"  {name}: {len(rows):,} unique rows ({skipped} invalid, {dupes} duplicates removed)")
    return rows


def download_slake(max_samples: "int | None" = None, image_quality: int = 85) -> list:
    return _download_hf_vqa("BoKelvin/SLAKE", "SLAKE", _SLAKE_IMG_FIELDS, max_samples, image_quality, log_interval=1000)


def download_vqa_rad(max_samples: "int | None" = None, image_quality: int = 85) -> list:
    return _download_hf_vqa("flaviagiammarino/vqa-rad", "VQA-RAD", _HF_VQA_IMG_FIELDS, max_samples, image_quality, log_interval=500)


def download_path_vqa(max_samples: "int | None" = None, image_quality: int = 85) -> list:
    return _download_hf_vqa("flaviagiammarino/path-vqa", "PathVQA", _HF_VQA_IMG_FIELDS, max_samples, image_quality, log_interval=2000)


def load_general_sample(parquet_path: str, n_samples: int, seed: int = 42) -> list:
    try:
        pf = pq.ParquetFile(parquet_path)
    except (FileNotFoundError, OSError):
        print(f"WARNING: general parquet not found at {parquet_path}, skipping mix", file=sys.stderr)
        return []

    available_columns = set(pf.schema_arrow.names)
    missing_columns = [
        col for col in _REQUIRED_GENERAL_COLUMNS if col not in available_columns
    ]
    if missing_columns:
        raise ValueError(
            "General parquet missing required columns: "
            f"{', '.join(missing_columns)}"
        )

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
    parser.add_argument("--max_slake",         type=int,   default=5000,
                        help="Max unique rows from SLAKE after dedup (default: 5000; set to 0 to skip)")
    parser.add_argument("--max_vqa_rad",       type=int,   default=3000,
                        help="Max unique rows from VQA-RAD after dedup (default: 3000; set to 0 to skip)")
    parser.add_argument("--max_path_vqa",      type=int,   default=5000,
                        help="Max unique rows from PathVQA after dedup (default: 5000; set to 0 to skip)")
    parser.add_argument(
        "--mix_general_ratio",
        type=_parse_mix_general_ratio,
        default=0.1,
        help=(
            "Fraction in [0.0, 1.0) of final rows from general parquet; "
            "0 disables mix (default: %(default)s)"
        ),
    )
    parser.add_argument("--general_parquet",   default="./dataset/sft_i2t.parquet")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--image_quality",
                        type=_parse_image_quality,
                        default=85,
                        help="JPEG quality 1–95 (lower = smaller files)")
    args = parser.parse_args()

    max_pmc = args.max_pmc_vqa if args.max_pmc_vqa > 0 else None
    medical_rows = download_pmc_vqa(max_samples=max_pmc, seed=args.seed, image_quality=args.image_quality)

    for attr, fn in (
        ("max_slake",    download_slake),
        ("max_vqa_rad",  download_vqa_rad),
        ("max_path_vqa", download_path_vqa),
    ):
        limit = getattr(args, attr)
        if limit != 0:
            medical_rows.extend(fn(
                max_samples=limit if limit > 0 else None,
                image_quality=args.image_quality,
            ))

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
