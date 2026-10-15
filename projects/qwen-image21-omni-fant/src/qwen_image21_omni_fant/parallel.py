"""Backend-neutral tensor transforms used by FANT sequence parallelism."""

from __future__ import annotations

from typing import Any


def shard_sequence(tensor: Any, rank: int, world_size: int, dim: int = 1) -> Any:
    """Return one equal contiguous sequence shard."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if tensor.shape[dim] % world_size:
        raise ValueError("sequence length must be divisible by world_size")
    return tensor.chunk(world_size, dim=dim)[rank].contiguous()


def unshard_sequence(shards: list[Any], dim: int = 1) -> Any:
    """Concatenate equal sequence shards without binding this package to torch at import."""
    if not shards:
        raise ValueError("at least one shard is required")
    import torch

    return torch.cat(shards, dim=dim).contiguous()


def all_to_all_4d(
    tensor: Any,
    group: Any,
    *,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    dist_module: Any | None = None,
) -> Any:
    """Ulysses 4D reshape plus all-to-all using the active PyTorch backend.

    Supported directions are sequence-sharded to head-sharded (2 -> 1), and
    the inverse (1 -> 2). The implementation intentionally uses only
    ``all_to_all_single`` so FHCCL can provide the transport.
    """
    if tensor.dim() != 4:
        raise ValueError(f"expected a 4D tensor, got shape {tuple(tensor.shape)}")
    if dist_module is None:
        import torch.distributed as dist_module
    world_size = dist_module.get_world_size(group)
    if world_size == 1:
        return tensor.contiguous()

    batch, seq, heads, head_dim = tensor.shape
    if (scatter_dim, gather_dim) == (2, 1):
        if heads % world_size:
            raise ValueError("head count must be divisible by world_size")
        local_heads = heads // world_size
        send = tensor.reshape(batch, seq, world_size, local_heads, head_dim).transpose(0, 2).contiguous()
        recv = send.new_empty(send.shape)
        dist_module.all_to_all_single(recv, send, group=group)
        return recv.reshape(seq * world_size, batch, local_heads, head_dim).transpose(0, 1).contiguous()

    if (scatter_dim, gather_dim) == (1, 2):
        if seq % world_size:
            raise ValueError("sequence length must be divisible by world_size")
        local_seq = seq // world_size
        send = (
            tensor.reshape(batch, world_size, local_seq, heads, head_dim)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(world_size, heads, local_seq, batch, head_dim)
        )
        recv = send.new_empty(send.shape)
        dist_module.all_to_all_single(recv, send, group=group)
        return recv.reshape(heads * world_size, local_seq, batch, head_dim).transpose(0, 2).contiguous()

    raise ValueError("supported directions are (2, 1) and (1, 2)")

