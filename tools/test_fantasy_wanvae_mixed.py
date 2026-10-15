import ast
import contextlib
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("fantasy_wanvae_mixed.py")


class FakeTensor:
    def __init__(self, dtype="fp32"):
        self.dtype = dtype
        self.device = type("Device", (), {"type": "fant"})()

    def float(self):
        return FakeTensor("fp32")

    def half(self):
        return FakeTensor("fp16")

    def __add__(self, other):
        assert self.dtype == other.dtype == "fp32"
        return FakeTensor("fp32")


class FakeTorch:
    float16 = "fp16"

    def __init__(self):
        self.autocast_enabled = False

    @contextlib.contextmanager
    def autocast(self, *, device_type, dtype=None, enabled=True):
        previous = self.autocast_enabled
        self.autocast_enabled = enabled
        try:
            yield
        finally:
            self.autocast_enabled = previous


class FakeLayer:
    def __init__(self, name, torch, calls, output_dtype="fp32"):
        self.name = name
        self.torch = torch
        self.calls = calls
        self.output_dtype = output_dtype

    def __call__(self, value):
        self.calls.append((self.name, value.dtype, self.torch.autocast_enabled))
        return FakeTensor(self.output_dtype if not self.torch.autocast_enabled else "fp16")


class ConvOnlyMixedPrecisionTest(unittest.TestCase):
    def test_only_convolutions_receive_fp16_inputs_without_autocast(self):
        tree = ast.parse(SOURCE.read_text())
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "residual_block_forward_fp16_convs"
        )
        module = ast.Module(body=[fn], type_ignores=[])
        ast.fix_missing_locations(module)

        fake_torch = FakeTorch()
        namespace = {"torch": fake_torch, "feat_cache": None, "feat_idx": None}
        exec(compile(module, str(SOURCE), "exec"), namespace)

        calls = []
        block = type("Block", (), {})()
        block.conv_shortcut = FakeLayer("shortcut", fake_torch, calls)
        block.norm1 = FakeLayer("norm1", fake_torch, calls)
        block.nonlinearity = FakeLayer("activation", fake_torch, calls)
        block.conv1 = FakeLayer("conv1", fake_torch, calls)
        block.norm2 = FakeLayer("norm2", fake_torch, calls)
        block.dropout = FakeLayer("dropout", fake_torch, calls)
        block.conv2 = FakeLayer("conv2", fake_torch, calls)

        output = namespace["residual_block_forward_fp16_convs"](
            block, FakeTensor("fp32")
        )

        self.assertEqual(output.dtype, "fp32")
        by_name = {name: (dtype, enabled) for name, dtype, enabled in calls}
        self.assertEqual(by_name["shortcut"], ("fp32", False))
        self.assertEqual(by_name["norm1"], ("fp32", False))
        self.assertEqual(by_name["activation"], ("fp32", False))
        self.assertEqual(by_name["conv1"], ("fp16", False))
        self.assertEqual(by_name["norm2"], ("fp32", False))
        self.assertEqual(by_name["dropout"], ("fp32", False))
        self.assertEqual(by_name["conv2"], ("fp16", False))


if __name__ == "__main__":
    unittest.main()
