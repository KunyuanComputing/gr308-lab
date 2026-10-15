import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "run_fant_graph_baseline.sh"


class FantGraphBaselineScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_uses_same_frozen_model_and_runtime(self):
        self.assertIn('MODEL_HOST="/data/models/Qwen3.8-27B"', self.text)
        self.assertIn('IMAGE="fantasy/vllm:1.1.0.20-py310-ubuntu22.04"', self.text)
        self.assertIn('VISIBLE_DEVICES="0,2,4,6"', self.text)
        self.assertIn('--tensor-parallel-size 4', self.text)
        self.assertIn('--dtype bfloat16', self.text)
        self.assertIn('--max-model-len 32768', self.text)
        self.assertIn('--gpu-memory-utilization 0.40', self.text)
        self.assertIn('--block-size 800', self.text)

    def test_enables_graph32_without_eager_or_graph64(self):
        expected = r"--compilation-config '\{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":\[1,2,4,8,16,32\]\}'"
        self.assertRegex(self.text, expected)
        # The sole occurrence is the runtime assertion that rejects eager mode.
        self.assertEqual(self.text.count("--enforce-eager"), 1)
        self.assertNotIn("cudagraph_capture_sizes\\\":[1,2,4,8,16,32,64]", self.text)

    def test_reproduces_rapid_formal_and_stability_protocol(self):
        self.assertIn('CONCURRENCIES=(1 8 16 32 64)', self.text)
        self.assertIn('RAPID_REQUESTS=(8 16 32 64 128)', self.text)
        self.assertIn('FORMAL_REQUESTS=(12 16 32 64 128)', self.text)
        self.assertIn('FORMAL_REPEATS=(r1 r3 r5)', self.text)
        self.assertRegex(self.text, r'run_case[^\n]+128[^\n]+64')
        self.assertRegex(self.text, r'run_case[^\n]+1024[^\n]+256')
        self.assertRegex(self.text, r'run_case[^\n]+128[^\n]+64[^\n]+1024')

    def test_has_correctness_telemetry_resume_and_cleanup(self):
        for token in (
            "correctness-pre",
            "correctness-post",
            "--query-gpu timestamp,index,pci.bus_id",
            "STATUS.events.jsonl",
            "case_complete",
            "trap cleanup EXIT",
            "docker rm -f",
            "aggregate_graph_baseline.py",
            "sentinel_screen.py",
        ):
            self.assertIn(token, self.text)

    def test_adaptive_formal_repeats_match_frozen_plan(self):
        self.assertIn("ADAPTIVE_REPEATS=(r2 r4)", self.text)
        self.assertIn("formal_needs_another_block", self.text)

    def test_sentinel_runner_owns_its_output_directory(self):
        function = self.text.split("run_sentinels() {", 1)[1].split("\n}", 1)[0]
        self.assertNotIn('mkdir -p "${output}"', function)

    def test_interrupt_trap_exits_after_cleanup(self):
        self.assertIn("trap cleanup EXIT", self.text)
        self.assertIn("trap 'exit 130' INT TERM", self.text)

    def test_c64_is_explicitly_labelled_outside_capture_bucket(self):
        self.assertIn("C64 is outside the frozen graph capture buckets", self.text)


if __name__ == "__main__":
    unittest.main()
