"""SGLang diffusion platform contract for Fantasy GR308 devices."""

import os
from functools import lru_cache
from typing import Any

import torch
import torch_fant  # noqa: F401 -- registers the PrivateUse1 backend as ``fant``

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)


class FantasyDiffusionPlatform(Platform):
    """Minimal fail-closed platform needed by the MiniMax H3 pipeline."""

    _enum = PlatformEnum.OOT
    device_name = "fant"
    device_type = "fant"
    dispatch_key = "PrivateUse1"
    device_control_env_var = "CUDA_VISIBLE_DEVICES"
    simple_compile_backend = "eager"
    supported_quantization = ["kitchen_w4a8"]

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device("fant", int(os.environ.get("LOCAL_RANK", "0")))

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("fant", local_rank)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        device = torch.device(device)
        if device.type != "fant":
            raise ValueError(
                "FantasyDiffusionPlatform refuses a non-Fantasy device"
            )
        torch.fant.set_device(device)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.fant.get_device_name(device_id))

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return int(torch.fant.get_device_properties(device_id).total_memory)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        properties = torch.fant.get_device_properties(device_id)
        uuid = getattr(properties, "uuid", None)
        if uuid is None:
            raise NotImplementedError("Fantasy device UUID is not exposed")
        return str(uuid)

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        del device_id
        return None

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        return float(torch.fant.memory_allocated(device))

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if distributed:
            raise NotImplementedError(
                "Distributed Fantasy memory admission is not validated"
            )
        if empty_cache:
            torch.fant.empty_cache()
        index = torch.fant.current_device() if device_id is None else device_id
        free_memory, _ = torch.fant.mem_get_info(index)
        return float(free_memory) / (1 << 30)

    @classmethod
    def get_torch_distributed_backend_str(cls) -> str:
        return "fhccl"

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        del head_size
        if selected_backend not in (None, AttentionBackendEnum.TORCH_SDPA):
            raise NotImplementedError("GR308 H3 permits Torch SDPA only")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported H3 attention dtype: {dtype}")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa."
            "SDPABackend"
        )

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        del enforce_eager
        return False

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        return False

    @classmethod
    def verify_quantization(cls, quant: str | None) -> None:
        if quant not in (None, "none", "unquantized", "kitchen_w4a8"):
            raise ValueError(f"{quant} quantization is not supported on GR308 H3")
