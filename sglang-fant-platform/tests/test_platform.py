# SPDX-License-Identifier: Apache-2.0

import unittest

import torch


class TestFantasySRTPlatform(unittest.TestCase):
    def test_platform_delegates_device_interface_to_torch_fant(self):
        from sglang_fant_platform import FantasySRTPlatform, activate

        platform = FantasySRTPlatform()
        self.assertEqual(
            activate(), "sglang_fant_platform:FantasySRTPlatform"
        )
        self.assertTrue(platform.is_out_of_tree())
        self.assertEqual(platform.device_name, "fant")
        self.assertEqual(platform.device_type, "fant")
        # SGLang's serving resolution strips an optional index with .split(),
        # so the platform hook must return the canonical device string.
        self.assertEqual(platform.get_device(0), "fant:0")
        self.assertEqual(platform.get_torch_distributed_backend_str(), "fhccl")
        self.assertEqual(platform.get_default_attention_backend(), "torch_native")
        self.assertTrue(platform.supports_fla())
        self.assertEqual(platform.get_mha_kv_pool_cls().__name__, "MHATokenToKVPool")
        self.assertEqual(
            platform.get_paged_allocator_cls().__name__,
            "PagedTokenToKVPoolAllocator",
        )
        device_module = torch.get_device_module(torch.device("fant"))
        self.assertIs(device_module.StreamContext, device_module.stream)
        self.assertEqual(platform.get_torch_library_dispatch_key(), "PrivateUse1")
        free_bytes, total_bytes = platform.get_available_memory(0)
        self.assertGreater(free_bytes, 0)
        self.assertGreaterEqual(total_bytes, free_bytes)


if __name__ == "__main__":
    unittest.main()
