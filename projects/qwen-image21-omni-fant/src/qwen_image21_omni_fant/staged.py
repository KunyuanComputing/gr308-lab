"""Small step-execution protocol adapted from vLLM-Omni.

The adapter owns no model code. It gives the existing patched Diffusers
pipeline explicit phase boundaries, which lets a launcher place or time each
phase independently on FANT devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GenerationState:
    latents: Any
    timesteps: Any
    scheduler: Any
    step_index: int = 0
    prompt_embeds: Any = None
    prompt_embeds_mask: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.step_index >= len(self.timesteps)

    @property
    def current_timestep(self) -> Any:
        if self.done:
            raise IndexError("generation already completed")
        return self.timesteps[self.step_index]


class StagedDiffusionRunner:
    """Execute a diffusion pipeline through Omni-compatible phase boundaries."""

    def __init__(
        self,
        *,
        prepare: Callable[..., GenerationState],
        predict_noise: Callable[[GenerationState], Any],
        scheduler_step: Callable[[GenerationState, Any], Any],
        decode: Callable[[GenerationState], Any],
    ) -> None:
        self._prepare = prepare
        self._predict_noise = predict_noise
        self._scheduler_step = scheduler_step
        self._decode = decode

    def prepare_encode(self, **kwargs: Any) -> GenerationState:
        state = self._prepare(**kwargs)
        if not isinstance(state, GenerationState):
            raise TypeError("prepare must return GenerationState")
        return state

    def denoise_step(self, state: GenerationState) -> Any:
        if state.done:
            raise RuntimeError("cannot denoise a completed generation")
        return self._predict_noise(state)

    def step_scheduler(self, state: GenerationState, noise_pred: Any) -> None:
        state.latents = self._scheduler_step(state, noise_pred)
        state.step_index += 1

    def post_decode(self, state: GenerationState) -> Any:
        if not state.done:
            raise RuntimeError("cannot decode before all scheduler steps complete")
        return self._decode(state)

    def run(self, **kwargs: Any) -> Any:
        state = self.prepare_encode(**kwargs)
        while not state.done:
            noise = self.denoise_step(state)
            self.step_scheduler(state, noise)
        return self.post_decode(state)

