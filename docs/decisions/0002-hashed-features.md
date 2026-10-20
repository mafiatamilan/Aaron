# 0002 — Hashed n-gram features, not a learned tokenizer

**Status:** accepted

## Context

A 14-intent classifier over free-form English needs features that survive
paraphrase, typos and ASR errors, and that can be reproduced exactly.

## Decision

Hash word unigrams, word bigrams and padded character trigrams with FNV-1a 64
into 10,240 buckets.

## Reasoning

* **No unfitted vocabulary, therefore no leakage.** A fitted tokenizer is fitted
  on the training split and can encode test-set statistics. A hash cannot.
* **Reproducible.** `DefaultHasher` explicitly does not guarantee stability
  across Rust versions; FNV-1a does. A feature map that shifted between
  compiler versions would silently invalidate every checkpoint.
* **Cheap robustness.** Character trigrams give tolerance to typos and ASR
  errors at zero parameter cost.

## Consequences

* ~8% bucket collisions across the dataset's roughly 2000 distinct features.
  Collisions add noise but cannot leak information across splits.
* No morphological generalisation: `status` and `statuses` share trigrams but
  not the unigram.
* The embedding table is 86% of the parameter budget, which is the correct
  trade for a bag-of-features text model at this size.
