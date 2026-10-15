import pytest

from scripts.fant_sdpa_bench import (
    compare,
    main,
    parse_config,
    passes_gates,
    percentile_summary,
)


def test_compare_is_finite_and_passes_accuracy_and_speed_gates():
    metrics = compare([1.0, 2.0], [1.005, 1.995])

    assert metrics["finite"] is True
    assert metrics["mae"] == pytest.approx(0.005)
    assert metrics["max_abs"] == pytest.approx(0.005)
    assert metrics["exact_fraction"] == 0.0
    assert passes_gates(metrics, candidate_median=0.90, reference_median=1.0)


def test_compare_handles_multidimensional_attention_tensors():
    torch = pytest.importorskip("torch")
    reference = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 2, 4)
    candidate = reference.clone()
    candidate[0, 1, 0, 2] += 0.25

    metrics = compare(reference, candidate)

    assert metrics == {
        "finite": True,
        "mae": pytest.approx(0.25 / 16),
        "max_abs": pytest.approx(0.25),
        "exact_fraction": pytest.approx(15 / 16),
    }


def test_speed_gate_includes_exactly_five_percent_faster():
    metrics = {"finite": True, "mae": 0.0, "max_abs": 0.0, "exact_fraction": 1.0}

    assert not passes_gates(metrics, candidate_median=0.951, reference_median=1.0)
    assert passes_gates(metrics, candidate_median=0.95, reference_median=1.0)


def test_percentile_summary_uses_nearest_rank_percentiles():
    assert percentile_summary([0.5, 0.1, 0.4, 0.2, 0.3]) == {
        "samples": [0.5, 0.1, 0.4, 0.2, 0.3],
        "median": 0.3,
        "p10": 0.1,
        "p90": 0.5,
    }


@pytest.mark.parametrize(
    ("environment", "setting"),
    [
        ({"SEQ_LEN": "17", "HEADS": "16", "HEAD_DIM": "128"}, "SEQ_LEN"),
        ({"SEQ_LEN": "128", "HEADS": "8", "HEAD_DIM": "128"}, "HEADS"),
        ({"SEQ_LEN": "128", "HEADS": "16", "HEAD_DIM": "64"}, "HEAD_DIM"),
    ],
)
def test_environment_rejects_unsupported_attention_shapes(environment, setting):
    with pytest.raises(ValueError, match=setting):
        parse_config([], environment)


def test_cli_shape_overrides_are_validated():
    with pytest.raises(ValueError, match="SEQ_LEN"):
        parse_config(["--seq-len", "17"], {})


def test_valid_shape_is_available_to_future_hardware_runner():
    config = parse_config(
        ["--seq-len", "4096", "--heads", "16", "--head-dim", "128"], {}
    )

    assert (config.seq_len, config.heads, config.head_dim) == (4096, 16, 128)


def test_main_reports_that_hardware_runner_is_not_implemented():
    with pytest.raises(RuntimeError, match="hardware benchmark execution is not implemented"):
        main([])
