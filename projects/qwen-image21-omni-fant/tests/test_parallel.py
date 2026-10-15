import pytest

from qwen_image21_omni_fant.parallel import shard_sequence


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(values))

    def chunk(self, count, dim):
        assert dim == 1
        size = len(self.values) // count
        return [FakeTensor(self.values[i * size : (i + 1) * size]) for i in range(count)]

    def contiguous(self):
        return self


def test_sequence_shards_are_equal_and_rank_ordered():
    source = FakeTensor(range(8))
    assert shard_sequence(source, 0, 2).values == [0, 1, 2, 3]
    assert shard_sequence(source, 1, 2).values == [4, 5, 6, 7]


def test_sequence_shard_rejects_non_divisible_length():
    with pytest.raises(ValueError, match="divisible"):
        shard_sequence(FakeTensor(range(7)), 0, 2)

