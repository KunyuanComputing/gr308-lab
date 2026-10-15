#!/usr/bin/env python3
import math
import unittest

import sglang_aggregate as agg


class AggregateMathTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(agg.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(agg.percentile([1, 2, 3, 4], 0.95), 3.85)

    def test_three_repeat_stats_report_uncertainty(self):
        result = agg.summarize_repeats([6.0, 6.2, 6.1])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], 6.1)
        self.assertGreater(result["sample_std"], 0)
        self.assertEqual(len(result["descriptive_t95_ci"]), 2)
        self.assertLess(result["descriptive_t95_ci"][0], result["mean"])
        self.assertGreater(result["descriptive_t95_ci"][1], result["mean"])

    def test_stability_protocol_is_exact(self):
        valid = {"input_tokens": 128, "output_tokens": 64, "concurrency": 64, "requests": 1024}
        invalid = dict(valid, input_tokens=1024, output_tokens=256)
        self.assertTrue(agg.is_comparable_stability(valid))
        self.assertFalse(agg.is_comparable_stability(invalid))


if __name__ == "__main__":
    unittest.main()
