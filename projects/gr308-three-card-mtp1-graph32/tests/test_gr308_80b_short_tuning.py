import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "gr308_80b_short_tuning.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gr308_80b_short_tuning", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_matrix_matches_approved_80b_short_design():
    module = load_module()
    specs = module.initial_specs()

    assert [spec.name for spec in specs] == [
        "A-eager-no-mtp", "B-eager-mtp1", "C-graph32", "D-graph64",
    ]
    assert all(spec.model == "Qwen3-Next-80B-A3B-Instruct" for spec in specs)
    assert all(spec.devices == (4, 5, 6, 7) and spec.tp_size == 4 for spec in specs)
    assert all((spec.input_tokens, spec.output_tokens) == (1024, 1024) for spec in specs)
    assert all(spec.reserved_devices == () and spec.short for spec in specs)
    assert specs[0].mtp == 0 and specs[0].allow_unused_mtp and specs[0].enforce_eager
    assert specs[1].mtp == 1 and specs[1].enforce_eager
    assert specs[2].graph == (4, 8, 16, 32) and not specs[2].enforce_eager
    assert specs[3].graph == (4, 8, 16, 32, 64)


def test_rank_prefers_decode_tps_then_aggregate_inside_three_percent_band():
    module = load_module()
    rows = [
        {"name": "decode-top", "status": "passed", "c16_decode_tps": 20.0,
         "c16_tps": 150.0, "c1_decode_tps": 30.0, "c16_median_ttft_ms": 5000.0},
        {"name": "aggregate", "status": "passed", "c16_decode_tps": 19.6,
         "c16_tps": 160.0, "c1_decode_tps": 29.0, "c16_median_ttft_ms": 5500.0},
        {"name": "outside", "status": "passed", "c16_decode_tps": 18.0,
         "c16_tps": 999.0, "c1_decode_tps": 99.0, "c16_median_ttft_ms": 1.0},
    ]

    assert module.rank_candidates(rows)[0]["name"] == "aggregate"


def test_followups_are_short_and_never_construct_formal_work():
    module = load_module()
    graph = module.initial_specs()[2]
    followups = module.followup_specs(graph)

    assert [spec.name for spec in followups] == ["E-auto-kv", "F-batch4096"]
    assert followups[0].kv_bytes is None
    assert followups[1].batched_tokens == 4096
    assert all(spec.short for spec in followups)
    assert not hasattr(module, "formal_spec")
