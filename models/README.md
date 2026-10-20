# Model manifest

AARON ships **one** model: the custom intent classifier in `aaron-brain`. This
file records it and everything else AARON may optionally use, with sizes and
licences, so the parameter budget is never ambiguous.

**AARON never downloads any of these.** Each is installed by you, by your
distribution's package manager, or by a download you perform yourself.

---

## AARON's own model — covered by the 1,000,000 budget

| field | value |
|---|---|
| name | AARON custom intent classifier |
| architecture id | `aaron-intent-1m-v1:10240x84h754c14L96` |
| trainable parameters | **1,000,000** (verified by test and by `aaron model`) |
| storage | 1,000,000 × f32 ≈ 4.0 MB on disk |
| tensor bytes in RAM | 4.0 MB |
| framework | `candle-nn` 0.11, CPU backend |
| output classes | 14 (fixed by the `Intent` enum) |
| input features | hashed word unigrams, bigrams and char trigrams, 10,240 buckets |
| dtype | f32 throughout; no mixed precision |
| device | CPU; no CUDA backend compiled in |
| licence | MIT, as part of this repository |
| source | trained from `training/dataset/current.jsonl` |
| download | none — train it with `aaron brain train` |

Verified parameter breakdown:

| tensor | shape | parameters |
|---|---|---|
| `emb.weight` | 10240 × 84 | 860,160 |
| `ln1.weight` | 168 | 168 |
| `ln1.bias` | 168 | 168 |
| `fc1.weight` | 754 × 168 | 126,672 |
| `fc1.bias` | 754 | 754 |
| `ln2.weight` | 754 | 754 |
| `ln2.bias` | 754 | 754 |
| `fc2.weight` | 14 × 754 | 10,556 |
| `fc2.bias` | 14 | 14 |
| **total** | | **1,000,000** |

---

## Optional components — **not** covered by the budget

These are separate artefacts with their own licences and their own memory
requirements. None is installed by `aaron setup` or `install-arch.sh`.

### Speech recognition

Whisper models in ggml format, run by whisper.cpp. Sizes are approximate and
depend on quantisation.

| model | parameters | file size (fp16) | licence |
|---|---|---|---|
| `ggml-tiny.en` | 39 M | 75 MB | MIT (code); weights per upstream OpenAI release terms |
| `ggml-base.en` | 74 M | 142 MB | as above |
| `ggml-small.en` | 244 M | 466 MB | as above |
| `ggml-medium.en` | 769 M | 1.5 GB | as above |
| `ggml-large-v3` | 1.55 B | 3.1 GB | as above |

Install: `sudo pacman -S whisper.cpp`, download a model yourself, then set
`asr.model_path`. AARON verifies `asr.model_sha256` when it is set.

### Speech synthesis

| engine | model | size | licence |
|---|---|---|---|
| espeak-ng | none (formant synthesiser) | n/a | GPL-3.0 |
| Piper | VITS ONNX voice | ~60 MB per voice | MIT (code); per-voice terms |
| speech-dispatcher | none (routes to an installed engine) | n/a | GPL-3.0 |

### Wake word

**None shipped.** AARON matches the configured phrase against the ASR transcript
using token-overlap and edit distance. With no ASR backend installed, the
energy-burst strategy is available and is honestly labelled as
false-activation-prone.

A production wake-word model (openWakeWord, Porcupine, …) is a separate download
with its own licence and its own parameter count, and is not part of AARON's
budget.

---

## The budget, stated plainly

The **1,000,000 parameter budget applies to AARON's custom intent model and to
nothing else.**

AARON does not claim that Whisper, a TTS voice or a wake-word model fits within
it, because none of them is small enough to. The `aaron model` command prints
this separation so it cannot be misread:

```console
$ aaron model
custom intent model (AARON brain)
  trainable params   1000000
  ...
other models (NOT part of the 1,000,000 budget)
  wake word          none shipped; AARON matches the phrase against the ASR transcript
  speech recognition whisper.cpp, user-installed, externally managed
  speech synthesis   espeak-ng / piper / spd-say, user-installed
  AARON never downloads a model.
```

---

## Adding a model

If you add one, add a row here with its parameter count and licence, and say
whether it is inside or outside the budget. A model whose accounting is unclear
is worse than no model at all.