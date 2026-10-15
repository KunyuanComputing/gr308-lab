import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "gr308_35b_two_card_tps.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gr308_35b_two_card_tps", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lanes_match_approved_matrix():
    module = load_module()
    lanes = module.lane_specs()

    assert len(lanes["graph"]) == 4
    assert len(lanes["scheduler"]) == 4
    assert {spec.devices for spec in lanes["graph"]} == {(0, 1)}
    assert {spec.devices for spec in lanes["scheduler"]} == {(2, 3)}
    assert [spec.graph for spec in lanes["graph"]] == [
        (4, 8, 16, 32, 64),
        (1, 2, 4, 8, 16, 32, 64),
        (4, 8, 16, 32, 64, 128),
        (1, 2, 4, 8, 16, 32, 64, 128),
    ]
    assert [(spec.batched_tokens, spec.async_scheduling) for spec in lanes["scheduler"]] == [
        (2048, False), (2048, True), (1536, False), (2560, False),
    ]
    assert all(
        spec.mtp == 3 and spec.block_size == 1088 and spec.kv_bytes is None
        and spec.reserved_devices == () and spec.short
        for specs in lanes.values() for spec in specs
    )


def test_rank_uses_c16_band_then_c1_and_ttft():
    module = load_module()
    rows = [
        {"name": "raw-top", "status": "passed", "c16_tps": 200.0,
         "c1_tps": 40.0, "c16_median_ttft_ms": 5000.0},
        {"name": "balanced", "status": "passed", "c16_tps": 196.0,
         "c1_tps": 45.0, "c16_median_ttft_ms": 4000.0},
        {"name": "outside", "status": "passed", "c16_tps": 190.0,
         "c1_tps": 100.0, "c16_median_ttft_ms": 1000.0},
    ]

    assert [row["name"] for row in module.rank_candidates(rows)] == [
        "balanced", "raw-top", "outside",
    ]


def test_cross_and_formal_specs_preserve_tuning_and_change_placement():
    module = load_module()
    source = module.lane_specs()["graph"][0]
    crossed = module.cross_spec(source, "cross-g-base")
    formal = module.formal_spec(source)

    assert crossed.devices == (2, 3)
    assert crossed.port == 22110
    assert crossed.container_name == "gr308-35b-tps-cross"
    assert crossed.graph == source.graph
    assert formal.devices == (0, 1)
    assert formal.container_name == "gr308-35b-tps-formal"
    assert formal.short is False
