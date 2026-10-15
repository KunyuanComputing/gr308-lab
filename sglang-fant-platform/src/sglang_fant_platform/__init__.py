# SPDX-License-Identifier: Apache-2.0
"""SGLang platform-interface adapter for Fantasy accelerators."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import DeviceCapability, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


def _fant():
    """Return the vendor device module after its PyTorch extension is loaded."""
    import torch_fant

    # torch.get_device_module(torch.device("fant")) returns the top-level
    # extension. SGLang calls its CUDA-compatible StreamContext constructor,
    # while the vendor exposes the identical context through stream().
    if not hasattr(torch_fant, "StreamContext"):
        torch_fant.StreamContext = torch_fant.stream
    return torch.fant


def activate() -> str | None:
    """Activate only when the vendor runtime reports usable hardware."""
    try:
        if not _fant().is_available():
            return None
    except (ImportError, RuntimeError, AttributeError):
        return None
    return "sglang_fant_platform:FantasySRTPlatform"


class FantasySRTPlatform(SRTPlatform):
    """Map SGLang's device interface onto ``torch.fant``.

    This adapter deliberately contains no kernels, scheduler policy, cache
    policy, precision changes, or graph implementation.
    """

    _enum = PlatformEnum.OOT
    device_name = "fant"
    device_type = "fant"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(_fant().get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(_fant().max_memory_allocated(device))

    def get_device(self, device_id: int = 0) -> str:
        return f"fant:{device_id}"

    def set_device(self, device: "torch.device") -> None:
        _fant().set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(_fant().get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        properties = _fant().get_device_properties(device_id)
        return str(getattr(properties, "uuid", f"fant:{device_id}"))

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        major, minor = _fant().get_device_capability(device_id)
        return DeviceCapability(int(major), int(minor))

    def empty_cache(self) -> None:
        _fant().empty_cache()

    def synchronize(self) -> None:
        _fant().synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        free_bytes, total_bytes = _fant().mem_get_info(device_id)
        return int(free_bytes), int(total_bytes)

    def is_pin_memory_available(self, device=None) -> bool:
        return device is None or str(device) != "cpu"

    def get_torch_distributed_backend_str(self) -> str:
        return "fhccl"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        super().seed_everything(seed)
        if seed is not None:
            _fant().manual_seed_all(seed)

    def get_default_attention_backend(self) -> str:
        return "torch_native"

    def get_mha_kv_pool_cls(self) -> type:
        # The upstream generic pool allocates through the supplied torch.device
        # and therefore works unchanged with the registered PrivateUse1 device.
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        return PagedTokenToKVPoolAllocator

    def supports_fla(self) -> bool:
        # The GR308 runtime ships a vendor Triton backend; this capability is
        # validated by tests/triton_smoke.py before enabling SGLang's FLA path.
        return True

    def get_dispatch_key_name(self) -> str:
        return "fant"

    def get_torch_library_dispatch_key(self) -> str:
        return "PrivateUse1"
