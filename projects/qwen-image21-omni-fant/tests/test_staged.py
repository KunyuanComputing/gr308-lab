import pytest

from qwen_image21_omni_fant.staged import GenerationState, StagedDiffusionRunner


def test_staged_runner_executes_each_step_then_decodes():
    events = []

    runner = StagedDiffusionRunner(
        prepare=lambda seed: GenerationState(latents=seed, timesteps=[3, 2, 1], scheduler=object()),
        predict_noise=lambda state: events.append(("denoise", state.current_timestep)) or 2,
        scheduler_step=lambda state, noise: events.append(("schedule", state.current_timestep)) or state.latents + noise,
        decode=lambda state: events.append(("decode", state.latents)) or state.latents,
    )

    assert runner.run(seed=1) == 7
    assert events == [
        ("denoise", 3), ("schedule", 3),
        ("denoise", 2), ("schedule", 2),
        ("denoise", 1), ("schedule", 1),
        ("decode", 7),
    ]


def test_decode_requires_completed_state():
    runner = StagedDiffusionRunner(prepare=lambda: None, predict_noise=lambda state: None, scheduler_step=lambda state, noise: None, decode=lambda state: None)
    with pytest.raises(RuntimeError):
        runner.post_decode(GenerationState(latents=0, timesteps=[1], scheduler=None))

