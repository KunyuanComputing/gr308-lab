"""Fail-closed evaluation of native Fantasy W4A8 dispatch evidence."""

BLOCKED_BACKEND_TOKENS = ("eager", "cpu", "cuda", "hip", "triton")


def evaluate_native_gate(report: dict) -> dict:
    """Return the report with an auditable native-dispatch decision."""

    reasons = []
    device_type = str(report.get("device_type", ""))
    output_device = str(report.get("output_device", ""))
    backend = str(report.get("selected_backend", "")).lower()
    relative_l2 = report.get("relative_l2")

    if device_type != "fant":
        reasons.append("input device is not Fantasy")
    if not output_device.startswith("fant:"):
        reasons.append("output device is not Fantasy")
    if report.get("finite") is not True:
        reasons.append("output is not finite")
    if not isinstance(relative_l2, (int, float)) or relative_l2 >= 0.03:
        reasons.append("relative L2 is missing or at least 0.03")

    has_privateuse_kernel = report.get("privateuse1_int_mm_kernel") is True
    identified_fant_backend = "fant" in backend and not any(
        token in backend for token in BLOCKED_BACKEND_TOKENS
    )
    if not (has_privateuse_kernel or identified_fant_backend):
        reasons.append("no native Fantasy W4A8/INT8 dispatch is proven")

    result = dict(report)
    result["native_gate_passed"] = not reasons
    result["gate_reason"] = "passed" if not reasons else "; ".join(reasons)
    return result

