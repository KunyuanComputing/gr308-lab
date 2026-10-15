#!/usr/bin/env python3
"""Correctness and bandwidth probe for two or more FANT logical devices."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
import torch_fant  # noqa: F401


def sync() -> None:
    torch.fant.synchronize()


def timed(operation, warmup: int = 2, iterations: int = 5) -> float:
    for _ in range(warmup):
        operation()
    sync()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    sync()
    return (time.perf_counter() - start) / iterations


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.fant.set_device(local_rank)
    device = torch.device(f"fant:{local_rank}")
    dist.init_process_group("fhccl")

    results: dict[str, object] = {"rank": rank, "local_rank": local_rank, "world_size": world}

    value = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(value)
    results["all_reduce_correct"] = value.item() == world * (world + 1) / 2

    value.fill_(rank)
    dist.broadcast(value, src=0)
    results["broadcast_correct"] = value.item() == 0

    gathered = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(world)]
    dist.all_gather(gathered, torch.tensor([rank], dtype=torch.int64, device=device))
    results["all_gather_correct"] = [int(x.item()) for x in gathered] == list(range(world))

    # FHCCL 1.0 has no alltoall_base implementation. Keep the unsupported
    # operation out of this process because the backend also aborts from its
    # allocator during exception cleanup. The isolated capability test records
    # that failure separately in the run log.
    results["all_to_all_correct"] = False
    results["all_to_all_reason"] = "fhccl does not support alltoall_base"

    for mib in (4, 32, 64):
        count = mib * 1024 * 1024 // 2
        payload = torch.ones(count, dtype=torch.bfloat16, device=device)
        seconds = timed(lambda: dist.all_reduce(payload))
        # Ring algorithm bus bandwidth: payload * 2*(N-1)/N.
        bus_bytes = payload.numel() * payload.element_size() * (2 * (world - 1) / world)
        results[f"all_reduce_{mib}MiB_ms"] = round(seconds * 1000, 3)
        results[f"all_reduce_{mib}MiB_bus_GBs"] = round(bus_bytes / seconds / 1e9, 3)

        gather_input = torch.arange(count // world, dtype=torch.int32, device=device)
        gather_output = [torch.empty_like(gather_input) for _ in range(world)]
        seconds = timed(lambda: dist.all_gather(gather_output, gather_input))
        results[f"all_gather_{mib}MiB_total_ms"] = round(seconds * 1000, 3)
        results[f"all_gather_{mib}MiB_total_GBs"] = round(count * 4 / seconds / 1e9, 3)

    print("FHCCL_PROBE " + json.dumps(results, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    # torch_fant 0.18.0 can assert in allocator teardown after a multi-process
    # job even when every collective completed. The container is disposable;
    # bypass Python/C++ static destruction after flushing the evidence.
    os._exit(0)


if __name__ == "__main__":
    main()
