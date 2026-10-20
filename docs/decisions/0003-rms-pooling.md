# 0003 — RMS pooling, not max pooling

**Status:** accepted, after measurement

## Context

The architecture needs two pooling statistics concatenated to reach `2D` while
hitting exactly 1,000,000 parameters. Mean alone is the obvious first half.

## Decision

Mean and RMS. Not max.

## Reasoning

Measured test accuracy on the shipped dataset:

| second statistic | test accuracy |
|---|---|
| max, scale 1.0 / 0.5 / 0.25 | 0.067 |
| mean (max discarded) | 0.750 |
| **RMS** | **0.767** |

Max pooling picks one winning feature per dimension, so a single unusual word
decides the class. Its scale is an order of magnitude above the mean, and
LayerNorm over the concatenated 168-vector then shrinks the mean half until it
carries almost no signal. RMS is the same order of magnitude as the mean, so
both halves survive.

## Two bugs this experiment exposed

1. The masking sentinel for max had the **wrong sign**. Adding `+1e9` made
   `max` a constant; LayerNorm's scale invariance then washed out the entire
   second half. The model trained to 70% on what was effectively a mean-pooled
   network, and the code looked correct.
2. The sentinel **poisoned the gradient**: `-1e9 * mask` inside the autograd
   graph multiplied the incoming gradient by a billion and diverged on the
   first step. The sentinel is now built from a *detached* mask.

RMS also removes the sentinel entirely: masked positions are exactly zero, so
padding contributes nothing to either statistic.

## Consequences

One fewer moving part, and no masking sentinel to get wrong. The 168-wide
concatenation still forces a genuinely 2-D pooled representation.
