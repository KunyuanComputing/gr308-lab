# Qwen-Image-2.1 Omni FANT Port Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox to track progress.

**Goal:** Port the useful vLLM-Omni Qwen-Image-2.1 execution and parallel patterns onto the proven GR308 Diffusers/FANT runtime, then complete a short multi-chip generation test.

**Architecture:** Keep the working patched Diffusers model path and add a small FANT adapter around it. Split execution into prepare, denoise-step, scheduler-step, and decode phases; use FHCCL for rank coordination, Ulysses-style all-to-all only when the backend passes a correctness probe, and distributed VAE tiles as the first practical multi-chip acceleration path. Do not import CUDA-only vLLM-Omni runtime code.

**Tech Stack:** Python 3.10, PyTorch 2.7.1 with `torch_fant`, FHCCL, Diffusers, Qwen-Image-2.1 BF16, Docker.

---

## Task 1: Establish the FHCCL capability boundary

- [x] Add a two-rank probe for broadcast, all-reduce, all-gather and all-to-all correctness.
- [x] Record latency and effective bandwidth for useful payload sizes.
- [x] Run it on logical chips 0 and 1 without restarting the host.
- [x] Gate each parallel feature on its required collective.

## Task 2: Add the Omni-style staged runner

- [x] Add explicit `prepare_encode`, `denoise_step`, `step_scheduler`, and `post_decode` phases around the working Diffusers pipeline.
- [x] Preserve deterministic seeds, BF16 precision, eager SDPA and the existing FANT compatibility patches.
- [x] Make phase transitions testable without loading the full checkpoint.

## Task 3: Add FANT parallel primitives

- [x] Implement FHCCL initialization and logical-device assignment.
- [x] Implement the reshape plus `all_to_all_single` primitive used by Ulysses attention.
- [x] Add CPU tests for reshape semantics and validation failures.
- [x] Disable CUDA graphs, FlashAttention, fused CUDA all-to-all and FP8 paths.

## Task 4: Add distributed VAE tile decode

- [x] Assign VAE tiles across ranks and gather decoded tiles on rank 0.
- [x] Reuse the model's overlap and blend rules.
- [x] Validate tile assignment and stitching with synthetic tensors before model execution.

## Task 5: Integrate and run the short test

- [x] Create a new remote workspace and reuse the existing model checkpoint without modifying the known-good environment.
- [x] Run import, operator and two-rank communication smoke tests first.
- [x] Run a 512×512, two-step BF16 generation on two logical chips.
- [x] If stable, run a 1024×1024, four-step test and capture phase timings, peak memory, output hash and logs.
- [x] Document supported and blocked parallel modes based on measured evidence.
