import unittest

import torch
import torch_fant  # noqa: F401

from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from h3_sglang_fant.platform import FantasyDiffusionPlatform


class FantasyPlatformTests(unittest.TestCase):
    def test_fantasy_platform_contract(self):
        platform = current_platform

        self.assertIsInstance(platform, FantasyDiffusionPlatform)
        self.assertEqual(platform.device_type, "fant")
        self.assertEqual(platform.dispatch_key, "PrivateUse1")
        self.assertEqual(platform.get_local_torch_device().type, "fant")
        self.assertEqual(platform.get_torch_distributed_backend_str(), "fhccl")
        self.assertTrue(
            platform.get_attn_backend_cls_str(
                AttentionBackendEnum.TORCH_SDPA, 128, torch.bfloat16
            ).endswith("sdpa.SDPABackend")
        )

    def test_refuses_non_fant_device(self):
        with self.assertRaisesRegex(ValueError, "non-Fantasy"):
            FantasyDiffusionPlatform.set_device(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
