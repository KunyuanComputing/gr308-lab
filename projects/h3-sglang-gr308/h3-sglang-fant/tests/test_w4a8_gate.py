from h3_sglang_fant.w4a8_gate import evaluate_native_gate


def test_accepts_privateuse1_native_dispatch():
    report = {
        "device_type": "fant",
        "privateuse1_int_mm_kernel": True,
        "selected_backend": "fant",
        "output_device": "fant:0",
        "finite": True,
        "relative_l2": 0.01,
    }

    assert evaluate_native_gate(report)["native_gate_passed"] is True


def test_accepts_identified_fantasy_backend_without_aten_kernel():
    report = {
        "device_type": "fant",
        "privateuse1_int_mm_kernel": False,
        "selected_backend": "fant_w4a8",
        "output_device": "fant:0",
        "finite": True,
        "relative_l2": 0.01,
    }

    assert evaluate_native_gate(report)["native_gate_passed"] is True


def test_rejects_eager_fallback_even_when_numerically_correct():
    report = {
        "device_type": "fant",
        "privateuse1_int_mm_kernel": False,
        "selected_backend": "eager",
        "output_device": "fant:0",
        "finite": True,
        "relative_l2": 0.01,
    }

    result = evaluate_native_gate(report)

    assert result["native_gate_passed"] is False
    assert "native" in result["gate_reason"].lower()


def test_rejects_bad_numerics():
    report = {
        "device_type": "fant",
        "privateuse1_int_mm_kernel": True,
        "selected_backend": "fant",
        "output_device": "fant:0",
        "finite": True,
        "relative_l2": 0.08,
    }

    assert evaluate_native_gate(report)["native_gate_passed"] is False
