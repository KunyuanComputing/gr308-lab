"""Tile assignment and stitching helpers for distributed VAE decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class Tile:
    tile_id: int
    row: int
    col: int
    y: int
    x: int


def iter_rank_tiles(height: int, width: int, tile_size: int, overlap: int, rank: int, world_size: int) -> Iterator[Tile]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("require tile_size > overlap >= 0")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    stride = tile_size - overlap
    tile_id = 0
    for row, y in enumerate(range(0, height, stride)):
        for col, x in enumerate(range(0, width, stride)):
            if (tile_id + 1) % world_size == rank:
                yield Tile(tile_id, row, col, y, x)
            tile_id += 1


def stitch_tiles(rows: list[list[Any]], blend_extent: int, row_limit: int, blend_v, blend_h) -> Any:
    """Apply the same vertical/horizontal blending order used by Diffusers."""
    if not rows or not rows[0]:
        raise ValueError("tile grid cannot be empty")
    import torch

    output_rows = []
    for i, row in enumerate(rows):
        output_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent)
            output_row.append(tile[..., :row_limit, :row_limit])
        output_rows.append(torch.cat(output_row, dim=-1))
    return torch.cat(output_rows, dim=-2)

