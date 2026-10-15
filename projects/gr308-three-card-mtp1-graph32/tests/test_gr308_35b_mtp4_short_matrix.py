import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "gr308_35b_mtp4_short_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gr308_35b_mtp4_short_matrix", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_matches_approved_six_group_design():
    module = load_module()
    waves = module.wave_specs()
    specs = [spec for wave in waves for spec in wave]

    assert [spec.name for spec in specs] == [
        "A-mtp4-g64-b2048", "B-mtp4-g128-b2048",
        "C-mtp4-g64-b3072", "D-mtp4-g64-b4096",
        "E-mtp3-g64-b2048", "F-mtp3-g64-b4096",
    ]
    assert [len(wave) for wave in waves] == [4, 2]
    assert [spec.devices for spec in waves[0]] == [
        (0, 1), (2, 3), (4, 5), (6, 7),
    ]
    assert [spec.devices for spec in waves[1]] == [(0, 1), (2, 3)]
    assert [spec.mtp for spec in specs] == [4, 4, 4, 4, 3, 3]
    assert [spec.batched_tokens for spec in specs] == [2048, 2048, 3072, 4096, 2048, 4096]
    assert specs[0].graph == (4, 8, 16, 32, 64)
    assert specs[1].graph == (4, 8, 16, 32, 64, 128)


def test_every_candidate_is_comparable_short_tp2_bf16_work():
    module = load_module()
    specs = [spec for wave in module.wave_specs() for spec in wave]

    assert all(spec.model == "Qwen3.6-35B-A3B" for spec in specs)
    assert all(spec.tp_size == 2 and spec.short for spec in specs)
    assert all((spec.input_tokens, spec.output_tokens) == (3072, 1024) for spec in specs)
    assert all(spec.kv_bytes is None and spec.block_size == 1088 for spec in specs)
    assert all(not spec.async_scheduling for spec in specs)
    assert all(spec.reserved_devices == () for spec in specs)
    assert module.STARTUP_READY_TIMEOUT_S >= 600
    assert not hasattr(module, "formal_spec")


def test_custom_all_reduce_is_never_disabled():
    module = load_module()
    specs = [spec for wave in module.wave_specs() for spec in wave]

    for spec in specs:
        argv = module.matrix.serve_args(spec)
        assert "--disable-custom-all-reduce" not in argv


def test_rankings_keep_decode_and_aggregate_goals_separate():
    module = load_module()
    rows = [
        {"name": "decode", "status": "passed", "c1_decode_tps": 51.0,
         "c16_decode_tps": 13.2, "c16_tps": 150.0},
        {"name": "aggregate", "status": "passed", "c1_decode_tps": 47.0,
         "c16_decode_tps": 12.8, "c16_tps": 171.0},
    ]

    winners = module.select_winners(rows)
    assert winners["c1_decode"]["name"] == "decode"
    assert winners["c16_decode"]["name"] == "decode"
    assert winners["c16_aggregate"]["name"] == "aggregate"


def test_next_worker_starts_only_after_previous_worker_is_ready():
    module = load_module()
    specs = module.wave_specs()[0][:2]
    events = []

    class FakeProcess:
        def __init__(self, *, target, args, name):
            self.name = name

        def start(self):
            events.append(("start", self.name))

    def process_factory(**kwargs):
        return FakeProcess(**kwargs)

    def readiness_waiter(spec, process):
        events.append(("ready", spec.name))
        assert process.name == spec.name

    processes = module.start_wave_processes(
        specs, [], object(), process_factory=process_factory,
        readiness_waiter=readiness_waiter,
    )

    assert len(processes) == 2
    assert events == [
        ("start", specs[0].name),
        ("ready", specs[0].name),
        ("start", specs[1].name),
    ]
