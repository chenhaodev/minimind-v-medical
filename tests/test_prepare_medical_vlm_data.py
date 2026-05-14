import argparse
import io
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from trainer.trainer_utils import get_lr

from dataset.prepare_medical_vlm_data import (
    _parse_mix_general_ratio,
    _parse_image_quality,
    download_pmc_vqa,
    download_slake,
    download_vqa_rad,
    download_path_vqa,
    load_general_sample,
)


class TestGetLr(unittest.TestCase):
    def test_no_warmup_starts_at_full_lr(self):
        # step 0, no warmup: cosine starts at lr * (0.1 + 0.9) = lr
        self.assertAlmostEqual(get_lr(0, 1000, 1e-4, warmup_steps=0), 1e-4, places=8)

    def test_warmup_step_zero_returns_zero(self):
        self.assertAlmostEqual(get_lr(0, 1000, 1e-4, warmup_steps=100), 0.0, places=8)

    def test_warmup_reaches_full_lr_at_warmup_steps(self):
        self.assertAlmostEqual(get_lr(100, 1000, 1e-4, warmup_steps=100), 1e-4, places=8)

    def test_warmup_is_linear(self):
        lr = get_lr(50, 1000, 1e-4, warmup_steps=100)
        self.assertAlmostEqual(lr, 0.5e-4, places=8)

    def test_lr_decays_after_warmup(self):
        lr_at_warmup_end = get_lr(100, 1000, 1e-4, warmup_steps=100)
        lr_midway = get_lr(550, 1000, 1e-4, warmup_steps=100)
        lr_at_end = get_lr(1000, 1000, 1e-4, warmup_steps=100)
        self.assertGreater(lr_at_warmup_end, lr_midway)
        self.assertGreater(lr_midway, lr_at_end)

    def test_minimum_lr_is_ten_percent_of_peak(self):
        # at total_steps the cosine term is cos(π)=-1, so lr*(0.1+0.45*(1-1))=lr*0.1
        lr_min = get_lr(1000, 1000, 1e-4, warmup_steps=0)
        self.assertAlmostEqual(lr_min, 1e-4 * 0.1, places=10)

    def test_backward_compat_no_warmup_arg(self):
        # default warmup_steps=0 must match old behaviour
        old = 1e-4 * (0.1 + 0.45 * (1 + __import__("math").cos(__import__("math").pi * 300 / 1000)))
        self.assertAlmostEqual(get_lr(300, 1000, 1e-4), old, places=10)


class PrepareMedicalVlmDataTest(unittest.TestCase):
    def test_parse_mix_general_ratio_accepts_zero_and_values_below_one(self):
        self.assertEqual(_parse_mix_general_ratio("0"), 0.0)
        self.assertEqual(_parse_mix_general_ratio("0.1"), 0.1)
        self.assertEqual(_parse_mix_general_ratio("0.999"), 0.999)

    def test_parse_mix_general_ratio_rejects_values_outside_half_open_range(self):
        for value in ("-0.1", "1", "1.1", "not-a-float"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _parse_mix_general_ratio(value)

    def test_parse_image_quality_accepts_pillow_jpeg_range(self):
        self.assertEqual(_parse_image_quality("1"), 1)
        self.assertEqual(_parse_image_quality("85"), 85)
        self.assertEqual(_parse_image_quality("95"), 95)

    def test_parse_image_quality_rejects_values_outside_pillow_jpeg_range(self):
        for value in ("0", "96", "-1", "not-an-int"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _parse_image_quality(value)

    def test_load_general_sample_rejects_parquet_missing_required_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet") as parquet_file:
            table = pa.table({"conversations": [json.dumps([])]})
            pq.write_table(table, parquet_file.name)

            with self.assertRaisesRegex(
                ValueError,
                "missing required columns: image_bytes",
            ):
                load_general_sample(parquet_file.name, n_samples=1)

    def test_load_general_sample_reads_required_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet") as parquet_file:
            table = pa.table({
                "conversations": [json.dumps([]), json.dumps([])],
                "image_bytes": [b"image-a", b"image-b"],
            })
            pq.write_table(table, parquet_file.name)

            rows = load_general_sample(parquet_file.name, n_samples=1, seed=42)

        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {"conversations", "image_bytes"})


def _make_rgb_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(100, 150, 200))


def _make_slake_row(question: str, answer: str, img: "Image.Image | None" = None) -> dict:
    return {
        "img_content": img or _make_rgb_image(),
        "question": question,
        "answer": answer,
        "q_lang": "en",
    }


class TestDownloadSlake(unittest.TestCase):
    def _run(self, rows: list, **kwargs) -> list:
        """Run download_slake with a mocked HuggingFace dataset."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(rows))
        mock_ds.__len__ = MagicMock(return_value=len(rows))
        mock_ds.column_names = ["img_content", "question", "answer", "q_lang"]

        # The function does `from datasets import load_dataset` at call time,
        # so we inject a mock module into sys.modules before the import runs.
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            return download_slake(**kwargs)

    def test_returns_one_row_per_valid_sample(self):
        rows = [
            _make_slake_row("What organ is shown?", "liver"),
            _make_slake_row("Is there a tumor?", "yes"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 2)

    def test_each_row_has_conversations_and_image_bytes(self):
        rows = [_make_slake_row("What is this?", "lung")]
        result = self._run(rows)
        self.assertEqual(set(result[0].keys()), {"conversations", "image_bytes"})
        parsed = json.loads(result[0]["conversations"])
        self.assertIsInstance(parsed, list)
        self.assertTrue(any(m["role"] == "user" for m in parsed))
        self.assertTrue(any(m["role"] == "assistant" for m in parsed))

    def test_image_placeholder_present_in_user_turn(self):
        rows = [_make_slake_row("Any findings?", "normal")]
        result = self._run(rows)
        parsed = json.loads(result[0]["conversations"])
        user_content = next(m["content"] for m in parsed if m["role"] == "user")
        self.assertIn("<image>", user_content)

    def test_deduplicates_identical_question_answer_pairs(self):
        rows = [
            _make_slake_row("Same question?", "same answer"),
            _make_slake_row("Same question?", "same answer"),  # duplicate
            _make_slake_row("Same question?", "same answer"),  # duplicate
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_max_samples_caps_output(self):
        rows = [_make_slake_row(f"Q{i}?", f"A{i}") for i in range(10)]
        result = self._run(rows, max_samples=3)
        self.assertEqual(len(result), 3)

    def test_skips_rows_with_missing_image(self):
        rows_raw = [
            {"img_content": None, "question": "Q?", "answer": "A", "q_lang": "en"},
            _make_slake_row("Valid?", "yes"),
        ]
        result = self._run(rows_raw)
        self.assertEqual(len(result), 1)

    def test_skips_rows_with_empty_question_or_answer(self):
        rows_raw = [
            _make_slake_row("", "some answer"),
            _make_slake_row("Some question?", ""),
            _make_slake_row("Valid?", "yes"),
        ]
        result = self._run(rows_raw)
        self.assertEqual(len(result), 1)

    def test_image_bytes_is_valid_jpeg(self):
        rows = [_make_slake_row("Structure?", "kidney")]
        result = self._run(rows)
        img = Image.open(io.BytesIO(result[0]["image_bytes"]))
        self.assertEqual(img.format, "JPEG")


def _make_vqa_rad_row(question: str, answer: str, img: "Image.Image | None" = None) -> dict:
    return {
        "image": img or _make_rgb_image(),
        "question": question,
        "answer": answer,
        "answer_type": "OPEN",
    }


class TestDownloadVqaRad(unittest.TestCase):
    def _run(self, rows: list, **kwargs) -> list:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(rows))
        mock_ds.__len__ = MagicMock(return_value=len(rows))
        mock_ds.column_names = ["image", "question", "answer", "answer_type"]
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            return download_vqa_rad(**kwargs)

    def test_returns_one_row_per_valid_sample(self):
        rows = [
            _make_vqa_rad_row("What structure is visible?", "trachea"),
            _make_vqa_rad_row("Is there cardiomegaly?", "yes"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 2)

    def test_each_row_has_conversations_and_image_bytes(self):
        rows = [_make_vqa_rad_row("Abnormality present?", "no")]
        result = self._run(rows)
        self.assertEqual(set(result[0].keys()), {"conversations", "image_bytes"})

    def test_image_placeholder_present_in_user_turn(self):
        rows = [_make_vqa_rad_row("Location of finding?", "left lung")]
        result = self._run(rows)
        parsed = json.loads(result[0]["conversations"])
        user_content = next(m["content"] for m in parsed if m["role"] == "user")
        self.assertIn("<image>", user_content)

    def test_deduplicates_identical_question_answer_pairs(self):
        rows = [_make_vqa_rad_row("Same?", "yes")] * 4
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_max_samples_caps_output(self):
        rows = [_make_vqa_rad_row(f"Q{i}?", f"A{i}") for i in range(10)]
        result = self._run(rows, max_samples=4)
        self.assertEqual(len(result), 4)

    def test_skips_rows_with_missing_image(self):
        rows = [
            {"image": None, "question": "Q?", "answer": "A", "answer_type": "CLOSED"},
            _make_vqa_rad_row("Valid?", "yes"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_skips_rows_with_empty_question_or_answer(self):
        rows = [
            _make_vqa_rad_row("", "some answer"),
            _make_vqa_rad_row("Some question?", ""),
            _make_vqa_rad_row("Valid?", "yes"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_image_bytes_is_valid_jpeg(self):
        rows = [_make_vqa_rad_row("Finding?", "pleural effusion")]
        result = self._run(rows)
        img = Image.open(io.BytesIO(result[0]["image_bytes"]))
        self.assertEqual(img.format, "JPEG")


def _make_path_vqa_row(question: str, answer: str, img: "Image.Image | None" = None) -> dict:
    return {
        "image": img or _make_rgb_image(),
        "question": question,
        "answer": answer,
        "answer_type": "yes/no",
    }


class TestDownloadPathVqa(unittest.TestCase):
    def _run(self, rows: list, **kwargs) -> list:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(rows))
        mock_ds.__len__ = MagicMock(return_value=len(rows))
        mock_ds.column_names = ["image", "question", "answer", "answer_type"]
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            return download_path_vqa(**kwargs)

    def test_returns_one_row_per_valid_sample(self):
        rows = [
            _make_path_vqa_row("Is mitosis present?", "yes"),
            _make_path_vqa_row("What cell type is shown?", "lymphocyte"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 2)

    def test_each_row_has_conversations_and_image_bytes(self):
        rows = [_make_path_vqa_row("Tumor present?", "no")]
        result = self._run(rows)
        self.assertEqual(set(result[0].keys()), {"conversations", "image_bytes"})

    def test_image_placeholder_present_in_user_turn(self):
        rows = [_make_path_vqa_row("Staining type?", "H&E")]
        result = self._run(rows)
        parsed = json.loads(result[0]["conversations"])
        user_content = next(m["content"] for m in parsed if m["role"] == "user")
        self.assertIn("<image>", user_content)

    def test_deduplicates_identical_question_answer_pairs(self):
        rows = [_make_path_vqa_row("Benign?", "yes")] * 5
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_max_samples_caps_output(self):
        rows = [_make_path_vqa_row(f"Q{i}?", f"A{i}") for i in range(10)]
        result = self._run(rows, max_samples=3)
        self.assertEqual(len(result), 3)

    def test_skips_rows_with_missing_image(self):
        rows = [
            {"image": None, "question": "Q?", "answer": "yes", "answer_type": "yes/no"},
            _make_path_vqa_row("Valid?", "no"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_skips_rows_with_empty_question_or_answer(self):
        rows = [
            _make_path_vqa_row("", "yes"),
            _make_path_vqa_row("Necrosis?", ""),
            _make_path_vqa_row("Valid?", "yes"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_image_bytes_is_valid_jpeg(self):
        rows = [_make_path_vqa_row("Grade?", "low")]
        result = self._run(rows)
        img = Image.open(io.BytesIO(result[0]["image_bytes"]))
        self.assertEqual(img.format, "JPEG")


_DEFAULT_CHOICES = ("Liver", "Kidney", "Pancreas", "Spleen")


def _make_pmc_vqa_row(
    question: str,
    answer: str,
    choices: tuple = _DEFAULT_CHOICES,
    img: "Image.Image | None" = None,
) -> dict:
    return {
        "Question": question,
        "Answer": answer,
        "Choice A": choices[0] if len(choices) > 0 else "",
        "Choice B": choices[1] if len(choices) > 1 else "",
        "Choice C": choices[2] if len(choices) > 2 else "",
        "Choice D": choices[3] if len(choices) > 3 else "",
        "Figure": img or _make_rgb_image(),
    }


class TestDownloadPmcVqa(unittest.TestCase):
    def _run(self, rows: list, **kwargs) -> list:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(rows))
        mock_ds.__len__ = MagicMock(return_value=len(rows))
        mock_ds.column_names = ["Question", "Answer", "Choice A", "Choice B", "Choice C", "Choice D", "Figure"]
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            return download_pmc_vqa(**kwargs)

    def test_resolves_letter_answer_to_choice_text(self):
        rows = [_make_pmc_vqa_row("What organ is this?", "A")]
        result = self._run(rows)
        parsed = json.loads(result[0]["conversations"])
        assistant_content = next(m["content"] for m in parsed if m["role"] == "assistant")
        self.assertEqual(assistant_content, "Liver")

    def test_strips_choices_from_question(self):
        rows = [_make_pmc_vqa_row("What organ is this?", "B")]
        result = self._run(rows)
        parsed = json.loads(result[0]["conversations"])
        user_content = next(m["content"] for m in parsed if m["role"] == "user")
        self.assertNotIn("A.", user_content)
        self.assertNotIn("Liver", user_content)

    def test_non_letter_answer_kept_as_is(self):
        rows = [_make_pmc_vqa_row("What is shown?", "trachea")]
        result = self._run(rows)
        parsed = json.loads(result[0]["conversations"])
        assistant_content = next(m["content"] for m in parsed if m["role"] == "assistant")
        self.assertEqual(assistant_content, "trachea")

    def test_skips_rows_with_missing_image(self):
        rows = [
            {"Question": "Q?", "Answer": "A", "Choice A": "X", "Choice B": "", "Choice C": "", "Choice D": "", "Figure": None},
            _make_pmc_vqa_row("Valid?", "B"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_skips_rows_with_empty_question_or_answer(self):
        rows = [
            _make_pmc_vqa_row("", "A"),
            _make_pmc_vqa_row("Some question?", ""),
            _make_pmc_vqa_row("Valid?", "C"),
        ]
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_deduplicates_on_resolved_answer(self):
        rows = [_make_pmc_vqa_row("Same Q?", "A")] * 3
        result = self._run(rows)
        self.assertEqual(len(result), 1)

    def test_max_samples_caps_output(self):
        rows = [_make_pmc_vqa_row(f"Q{i}?", "A") for i in range(10)]
        result = self._run(rows, max_samples=3)
        self.assertEqual(len(result), 3)

    def test_image_bytes_is_valid_jpeg(self):
        rows = [_make_pmc_vqa_row("Structure?", "D")]
        result = self._run(rows)
        img = Image.open(io.BytesIO(result[0]["image_bytes"]))
        self.assertEqual(img.format, "JPEG")


if __name__ == "__main__":
    unittest.main()
