"""Measured communication capabilities used to gate parallel modes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FhcclCapabilities:
    broadcast: bool
    all_reduce: bool
    all_gather: bool
    all_to_all: bool

    @property
    def supports_ulysses(self) -> bool:
        return self.all_to_all

    @property
    def supports_distributed_vae_tiles(self) -> bool:
        return self.broadcast and self.all_reduce


GR308_FHCCL_018 = FhcclCapabilities(
    broadcast=True,
    all_reduce=True,
    all_gather=True,
    all_to_all=False,
)

