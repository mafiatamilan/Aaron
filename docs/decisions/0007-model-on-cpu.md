# 0007 — AARON's own model runs on CPU

**Status:** accepted

## Context

The target machine is an RTX 2050 with 4 GB of VRAM, and AARON is expected to
use the GPU where it helps.

## Decision

Candle's CPU backend for the intent model. No CUDA backend is compiled in.

## Reasoning

* At 1,000,000 parameters (4 MB of f32), the model is small enough that the
  host-to-device transfer dominates. Moving 4 MB per utterance to save a
  microsecond of matmul is a net loss.
* On a 4 GB card the compositor and the browser already hold most of the
  memory. AARON should not be competing for it with a model that gains nothing.
* One code path is easier to test, to reproduce and to reason about. GPU
  numerics introduce a second set of results to explain.

## Consequences

* `aaron doctor` reports `CUDA support compiled in: false`, which is the truth.
* If the model grows past roughly 100 M parameters, this decision should be
  revisited. It is documented in `docs/resources.md` as the trigger.
* whisper.cpp may still use the GPU, because that is the user's own
  configuration and a separate model. AARON never manages that VRAM itself.
