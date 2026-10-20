# 0005 — Ship a deterministic router, not a model-only product

**Status:** accepted

## Context

The custom model needs a trained checkpoint. Requiring one before AARON does
anything would make the first run a download, and AARON's promise is that
nothing is downloaded.

## Decision

`brain.backend = "auto"`: use the checkpoint when a promoted one exists,
otherwise a transparent keyword/grammar router.

## Reasoning

* `aaron ask "check system status"` works on a fresh install with no models, no
  network and no microphone.
* The fallback is reported honestly in every response (`classifier:
  "heuristic"`) and in `aaron doctor`, so it is never mistaken for the model.
* Every downstream component depends on the `IntentClassifier` trait, not on
  Candle, so the fallback costs nothing architecturally.

## Consequences

* The shipped model does not yet meet AARON's promotion thresholds, so in
  practice the deterministic router is what runs. This is stated in the README
  rather than hidden, and it is a dataset-size problem, not an architectural
  one.
* Two classifiers to keep in step. Both are covered by the same test shape, and
  `routes_the_canonical_examples` pins the router's behaviour.
