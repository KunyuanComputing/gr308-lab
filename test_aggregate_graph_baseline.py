import json
import tempfile
import unittest
from pathlib import Path

import aggregate_graph_baseline as aggregate


class AggregateGraphBaselineTest(unittest.TestCase):
    def test_case_entry_rejects_inexact_or_failed_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / "summary.json").write_text(json.dumps({
                "requests": 2,
                "concurrency": 1,
                "input_tokens": 128,
                "output_tokens": 64,
                "measurement_duration_s": 10,
                "raw_completed_output_tokens_per_s": 12.8,
                "statuses": {"ok": 1, "http_error": 1},
            }))
            (case / "client.rc").write_text("0\n")
            row = aggregate.case_entry("rapid", "rapid", case)
            self.assertEqual(row["ok"], 1)
            self.assertEqual(row["requests"], 2)
            self.assertFalse(aggregate.case_passed(row))

    def test_descriptive_stats_use_engine_blocks(self):
        result = aggregate.descriptive_stats([10.0, 12.0, 14.0])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 12.0)
        self.assertAlmostEqual(result["sample_std"], 2.0)
        self.assertIn("descriptive_t95_ci", result)


if __name__ == "__main__":
    unittest.main()
