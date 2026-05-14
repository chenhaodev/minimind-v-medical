import argparse
import json
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from dataset.prepare_medical_vlm_data import (
    _parse_mix_general_ratio,
    _parse_image_quality,
    load_general_sample,
)


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


if __name__ == "__main__":
    unittest.main()
