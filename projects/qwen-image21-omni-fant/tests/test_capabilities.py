from qwen_image21_omni_fant.capabilities import GR308_FHCCL_018


def test_measured_fhccl_capabilities_gate_parallel_modes():
    assert GR308_FHCCL_018.supports_distributed_vae_tiles
    assert not GR308_FHCCL_018.supports_ulysses

