import pytest

from qwen_image21_omni_fant.vae_tiles import iter_rank_tiles


def test_tile_assignment_is_complete_and_disjoint():
    rank0 = list(iter_rank_tiles(8, 8, 4, 1, 0, 2))
    rank1 = list(iter_rank_tiles(8, 8, 4, 1, 1, 2))
    ids0 = {tile.tile_id for tile in rank0}
    ids1 = {tile.tile_id for tile in rank1}
    assert not ids0 & ids1
    assert ids0 | ids1 == set(range(9))


def test_invalid_tile_geometry_rejected():
    with pytest.raises(ValueError):
        list(iter_rank_tiles(8, 8, 4, 4, 0, 2))
