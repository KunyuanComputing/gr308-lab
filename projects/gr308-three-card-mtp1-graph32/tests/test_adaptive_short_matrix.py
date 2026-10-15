import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "gr308_adaptive_short_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gr308_adaptive_short_matrix", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_seed_specs_match_requested_first_comparisons():
    module = load_module()

    assert module.initial_specs("35b")[0].graph == (4, 8, 16, 32)
    assert module.initial_specs("35b")[1].mtp == 3
    assert module.initial_specs("35b")[1].graph == (4, 8, 16, 32, 64)
    assert module.initial_specs("9b")[0].graph == (4, 8, 16, 32, 64)
    assert module.initial_specs("30b")[1].moe_backend == "triton"


def test_serve_args_encode_tuning_dimensions_without_disabling_all_reduce():
    module = load_module()
    spec = module.Spec(
        "probe", "Qwen3.5-9B", (2, 3), 22110, "8-15", 3,
        (4, 8, 16, 32, 64), 4096, 1088, None,
        moe_backend="triton", async_scheduling=True, tp_size=1, dp_size=2,
        fuse_gemm_comms=True, fuse_allreduce_rms=True, max_num_seqs=24,
        short=False,
    )
    args = module.serve_args(spec)

    assert "--disable-custom-all-reduce" not in args
    assert args[args.index("--max-num-batched-tokens") + 1] == "4096"
    assert args[args.index("--max-num-seqs") + 1] == "24"
    assert args[args.index("--block-size") + 1] == "1088"
    assert "--kv-cache-memory-bytes" not in args
    assert "--async-scheduling" in args
    assert "--no-async-scheduling" not in args
    assert args[args.index("--tensor-parallel-size") + 1] == "1"
    assert args[args.index("--data-parallel-size") + 1] == "2"
    assert args[args.index("--moe-backend") + 1] == "triton"
    compilation = module.json.loads(args[args.index("--compilation-config") + 1])
    assert compilation["pass_config"]["fuse_gemm_comms"] is True
    assert compilation["pass_config"]["fuse_allreduce_rms"] is True
    assert module._job(spec, Path("/tmp/results")).short is False


def test_choose_best_prefers_c16_then_c1_inside_three_percent_band():
    module = load_module()
    rows = [
        {"name": "a", "status": "passed", "c1_tps": 40.0, "c16_tps": 200.0},
        {"name": "b", "status": "passed", "c1_tps": 55.0, "c16_tps": 204.0},
        {"name": "failed", "status": "failed", "c1_tps": 999.0, "c16_tps": 999.0},
    ]

    assert module.choose_best(rows)["name"] == "b"


def test_80b_spec_builds_tp4_eager_1024_workload_without_disabling_custom_ar():
    module = load_module()
    spec = module.Spec(
        name="probe", model="Qwen3-Next-80B-A3B-Instruct",
        devices=(4, 5, 6, 7), port=22120, cpu_set="16-31", mtp=1,
        graph=(4, 8, 16, 32), batched_tokens=2048, block_size=1088,
        kv_bytes=module.FIXED_KV, tp_size=4, enforce_eager=True,
        container_name="gr308-80b-short", reserved_devices=(),
        input_tokens=1024, output_tokens=1024,
    )

    args = module.serve_args(spec)
    job = module._job(spec, Path("/tmp/results"))

    assert "--enforce-eager" in args
    assert "--compilation-config" not in args
    assert "--disable-custom-all-reduce" not in args
    assert args[args.index("--tensor-parallel-size") + 1] == "4"
    assert job.execution == "TP4"
    assert job.container_name == "gr308-80b-short"
    assert job.devices == (4, 5, 6, 7)
    assert (job.input_tokens, job.output_tokens) == (1024, 1024)
