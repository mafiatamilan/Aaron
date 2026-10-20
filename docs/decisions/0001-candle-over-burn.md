# 0001 — Candle, not Burn, for the intent model

**Status:** accepted

## Context

The custom intent model needs a Rust framework that can build a 1,000,000
parameter network, train it deterministically, save versioned checkpoints, and
enumerate its parameters exactly.

## Decision

Use `candle-core` / `candle-nn` (0.11).

## Reasoning

* **Dependency surface.** Candle pulls in gemm, rand, serde, half and memmap2.
  Burn pulls in a much larger stack, several of which are still pre-1.0. On a
  machine where the model is 4 MB, the framework's compile-time cost and
  transitive risk matter more than its flexibility.
* **Exact parameter counting.** Candle's `VarBuilder` builds from an explicit
  tensor map, so the parameter count is computed from the tensors the
  architecture actually declares. The budget test is a real check, not a
  restatement of a constant.
* **Checkpoint control.** Writing a custom container — manifest, per-tensor
  SHA-256, whole-block SHA-256 — was straightforward because AARON owns the
  tensor map end to end.
* **Determinism.** AARON supplies its own SplitMix64 for initialisation and its
  own shuffling, and CPU reductions are fixed-size. Two runs with the same seed
  produce bit-identical weights, which `training_is_deterministic` verifies.

## Consequences

* Candle's CPU allocator caches freed blocks, so releasing a model returns
  AARON's reference but does not promise the OS reclaims the pages. `aaron
  resources` reports measured RSS rather than a derived figure.
* GPU support exists in Candle but is not compiled in. At 1M parameters, the
  host-device transfer costs more than the matmul saves. If a GPU backend is
  added later, it is an optional feature and CPU stays the default.
