import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "gr308_parallel_goal.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gr308_parallel_goal", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_matrix_uses_three_cards_and_reserves_fourth():
    module = load_module()
    jobs = module.build_jobs(Path("/tmp/results"))

    assert [(job.model, job.devices, job.port, job.cpu_set) for job in jobs] == [
        ("Qwen3.6-35B-A3B", (0, 1), 22100, "0-7"),
        ("Qwen3.5-9B", (2, 3), 22110, "8-15"),
        ("Qwen3-30B-A3B", (4, 5), 22120, "16-23"),
    ]
    assert module.RESERVED_DEVICES == (6, 7)


def test_graph32_and_mtp_policy_are_exact():
    module = load_module()
    jobs = module.build_jobs(Path("/tmp/results"))

    for job in jobs:
        args = module.build_serve_args(job)
        assert "--disable-custom-all-reduce" not in args
        compilation = args[args.index("--compilation-config") + 1]
        assert '"cudagraph_capture_sizes": [32]' in compilation

        has_spec = "--speculative-config" in args
        if job.model in {"Qwen3.6-35B-A3B", "Qwen3.5-9B"}:
            assert has_spec
            assert '"num_speculative_tokens": 1' in args[args.index("--speculative-config") + 1]
        else:
            assert not has_spec
            assert "--enable-expert-parallel" in args


def test_short_tuning_job_records_multi_size_graph_and_request_plan():
    module = load_module()
    job = module.build_qwen35_short_job(Path("/tmp/results"), mtp_tokens=3)

    assert job.graph_capture_sizes == (4, 8, 16, 32)
    compilation = module.build_serve_args(job)[
        module.build_serve_args(job).index("--compilation-config") + 1
    ]
    assert '"cudagraph_capture_sizes": [4, 8, 16, 32]' in compilation
    assert module.short_request_count(1) == 16
    assert module.short_request_count(16) == 32


def test_mtp_profile_records_actual_depth():
    module = load_module()

    assert module.profile_name(0) == "baseline-no-native-mtp"
    assert module.profile_name(1) == "mtp1"
    assert module.profile_name(3) == "mtp3"


def test_registered_container_policy_rejects_unknown_container():
    module = load_module()
    jobs = module.build_jobs(Path("/tmp/results"))
    allowed = module.allowed_container_names(jobs)

    assert module.containers_allowed({jobs[0].container_name}, allowed)
    assert module.containers_allowed(allowed, allowed)
    assert not module.containers_allowed(allowed | {"unrelated"}, allowed)


def test_card_cleanup_is_scoped_and_reserved_card_stays_idle():
    module = load_module()
    memories = [540, 540, 39000, 39000, 39000, 39000, 540, 540]

    assert module.card_cleanup_ok(memories, (0, 1))
    assert module.card_cleanup_ok([533, 533, 39000, 39000, 39000, 39000, 533, 533], (0, 1))
    assert not module.card_cleanup_ok(memories, (2, 3))
    assert not module.card_cleanup_ok(memories[:-1] + [636], (0, 1))
    assert not module.card_cleanup_ok([499, 533, 39000, 39000, 39000, 39000, 533, 533], (0, 1))


def test_job_can_disable_reserved_device_check_for_coordinated_full_card_use():
    module = load_module()
    job = module.Job(
        "m", (4, 5, 6, 7), 22120, "16-31", Path("/tmp/o"),
        "c", 1, "TP4", reserved_devices=(),
    )

    assert job.reserved_devices == ()
    assert module.card_cleanup_ok([540] * 8, job.devices, job.reserved_devices)


def test_job_carries_exact_workload_shape():
    module = load_module()
    job = module.Job(
        "m", (4, 5, 6, 7), 22120, "16-31", Path("/tmp/o"),
        "c", 1, "TP4", input_tokens=1024, output_tokens=1024,
        allow_unused_mtp=True,
    )

    assert (job.input_tokens, job.output_tokens) == (1024, 1024)
    assert job.allow_unused_mtp is True


def test_all_core_health_requires_all_eight_devices_ok():
    module = load_module()
    healthy = "\n".join(
        f"Driver Status: [OK]\nDevice {index}:{128 + index} Status:[OK]"
        for index in range(8)
    )

    assert module.all_core_health_ok(healthy)
    assert not module.all_core_health_ok(healthy.replace("Device 5:133 Status:[OK]", "Device 5:133 Status:[Error]"))
