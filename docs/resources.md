# Memory and resource management

AARON treats memory as a budget, not as something to discover at 3am.

This document describes what is bounded, what is measured, what is degraded
when the machine is short of memory, and — importantly — what AARON
deliberately refuses to do to save memory.

Everything here is implemented in `crates/aaron-resources`, configured through
`[resources]` in `config/default.toml`, and inspected with:

```
aaron resources status     # measured, right now
aaron resources limits     # every configured budget
aaron resources stress     # repeatable workloads, stability verdict
aaron resources soak --seconds 60
```

---

## 1. The contract

Five guarantees, each enforced by code rather than by convention:

| Guarantee | Enforced by | Test |
|---|---|---|
| Every queue and retained buffer has a finite, configured bound | `ResourceConfig::validate` | `unsafe_resource_values_are_rejected` |
| A model is never released while a caller holds it | `ModelGuard` type | `a_model_in_use_is_never_unloaded` |
| Low-memory mode does not lose user data | `degradations_for_low_memory` | `low_memory_mode_does_not_affect_persistent_memory` |
| Repeated work does not leak | `aaron_resources::stress` | `repeated_allocation_is_stable` |
| Anything unmeasurable is reported as unmeasurable | `probe::gpu_memory` | `an_unavailable_gpu_never_reports_zero_as_a_measurement` |

### What AARON will never do to save memory

* **Never** kill another application. AARON does not own the rest of the
  desktop.
* **Never** abandon a database write in flight.
* **Never** discard user-approved persistent memory. Freeing RAM is not a
  reason to forget a fact the user explicitly asked AARON to keep. Only an
  explicit `aaron memory delete <id>` removes a stored fact.
* **Never** silently terminate the user's current task. Low-memory mode defers
  *new* background work; a tool already running is allowed to finish under its
  own timeout.

---

## 2. Configured budgets

`[resources]` in `config/default.toml`. Every value is validated at load time;
an unsafe value is a startup error, not a surprise at runtime.

| Setting | Default | Meaning |
|---|---|---|
| `low_memory_mode` | `true` | Enable the state machine at all |
| `monitor_interval_seconds` | `5` | Seconds between resource samples |
| `reserve_system_ram_mb` | `1024` | RAM AARON refuses to use, left for the desktop |
| `reserve_gpu_vram_mb` | `768` | VRAM AARON refuses to use, left for the compositor |
| `low_memory_free_percent` | `12` | Free-memory percentage that triggers degradation |
| `low_memory_recover_percent` | `20` | Percentage that clears it; **must** be higher |
| `max_audio_buffer_seconds` | `10` | Audio retained in RAM at once |
| `max_pending_requests` | `4` | Queued requests before new ones are refused |
| `max_conversation_turns_in_context` | `8` | Turns kept in the RAM context ring |
| `max_tool_output_bytes` | `65536` | Ceiling for one tool's output in RAM |
| `max_transcript_chars` | `12000` | Ceiling for a retained transcript |
| `allow_model_unloading` | `true` | Whether idle models may be released at all |

`[resources.models]` — `load_asr_on_demand`, `load_tts_on_demand`,
`unload_idle_models`, `idle_unload_seconds` (default 120).

`[resources.memory]` — `sqlite_cache_kib` (2048), `max_retrieved_memories`
(10), `max_memory_content_bytes` (16384), `max_learning_queue_items_in_memory`
(100).

### Hysteresis is mandatory

`low_memory_recover_percent` must be strictly greater than
`low_memory_free_percent`, and the config loader rejects it otherwise. On top of
that gap the manager requires `RECOVERY_SAMPLES_REQUIRED` (2) *consecutive*
samples that were themselves above the recovery threshold before it leaves
degraded mode.

Both defences are tested. `hovering_near_the_threshold_does_not_flap` alternates
200 samples either side of the trigger and asserts the mode changes exactly once.

---

## 3. Why the reserves exist

The RTX 2050 has 4 GB of VRAM, and AARON must not assume it has any of it:

* the compositor, the browser and the desktop already hold most of it;
* AARON's own CUDA backend is **not compiled in**, so AARON holds no CUDA
  context itself;
* the only GPU use AARON can cause is a child process the user chose to run
  (whisper.cpp), and that is their decision, made with their own model file.

`reserve_gpu_vram_mb` therefore exists so that even a generous whisper.cpp
configuration cannot be pointed at the whole card. `aaron resources status`
prints the card total, the amount used by *everything*, and the amount AARON
considers usable — labelled separately, because "AARON's VRAM" and "this
machine's VRAM" are different questions.

Similarly `reserve_system_ram_mb` leaves headroom for the desktop rather than
letting AARON be the reason the compositor is restarted.

---

## 4. Model lifecycle

`ModelSlot<T, F>` loads a model on first use and may release it when idle.

```
     acquire()                        drop(guard)
  ┌──────────────┐                 ┌──────────────┐
  │  no model    │  load           │  in use (n)  │
  │  users = 0   │ ──────────────► │  users = n   │
  └──────────────┘                 └──────────────┘
         ▲                                   │
         │ release_if_idle / release_now     │ only when users == 0
         │                                   ▼
         └──────────────────────────  idle, still resident
                                         │
                                         │ after idle_unload_seconds
                                         ▼
                                     released
```

The property that matters: **a model is never released while a handle is
alive**. `ModelGuard` holds the user count for exactly as long as the caller
needs the model, so "unload while in use" is not a race that can be lost — it
is not expressible.

A release attempt while the model is in use is refused and counted
(`unload_skipped_in_use`), so `aaron resources limits` can show that unloading
was skipped rather than silently lost.

A failed load leaves the slot empty and usable, increments `load_failures`, and
propagates the error. There is no half-built state to recover from.

---

## 5. Low-memory mode

When available RAM (or *measured* free VRAM) drops below
`low_memory_free_percent`, AARON:

1. refuses to start nonessential background work — `admits(true)` returns an
   error explaining why;
2. defers background training and indexing;
3. caps the audio buffer and discards stale audio rather than queueing it;
4. releases idle ASR/TTS models, when unloading is enabled;
5. keeps the conversation context at its configured bound and never grows it;
6. tells the user exactly which capabilities are degraded —
   `aaron resources status` lists them, and `set_activity` keeps reporting the
   *real* activity separately so the UI can show "listening (low memory)" rather
   than a bare error.

It does **not** drop stored memories, kill processes, or abort a running tool.

A GPU that cannot be measured is never treated as full: `GpuSample::unavailable`
reports `free_percent() == 100.0`, so an absent or unresponsive `nvidia-smi`
cannot spuriously trigger degradation.

---

## 6. Bounded audio

The capture callback writes fixed-size blocks into a bounded `sync_channel`.

```
capacity = clamp(max_audio_buffer_seconds × sample_rate × 4 / frame_bytes, 2, 4096)
```

At the default 10 s and 16 kHz mono `f32` that is 640 KiB — independent of how
slow the consumer is. When the queue is full the callback **drops the block
and increments a counter** rather than blocking the ALSA thread. Dropping is
the right trade: blocking the capture callback would stall the driver's ring
buffer and cause hardware-level glitches, which are worse than losing a frame.
The dropped count is surfaced by `aaron listen` so a stalled pipeline is visible
rather than silently truncating speech.

The callback never performs model inference, disk I/O or network access. It
decodes samples and offers a block, and that is all.

---

## 7. Conversational memory vs operating-system memory

These are different things and AARON keeps them apart.

**Durable, user-approved facts** live in SQLite. They are retrieved on demand
with a bounded query; the whole database is never loaded.

**The RAM context ring** holds at most `max_conversation_turns_in_context` turns
*and* at most `max_transcript_chars` characters. Both bounds apply, because one
enormous turn must not be able to blow a turn-count budget on its own.

When the ring wraps, evicted turns are folded into a summary that records the
ids of the turns it was built from:

```json
{
  "text": "#1 system_status: check system status | #7 system_status: how is the machine",
  "source_turn_ids": [1, 7],
  "intents": ["system_status"]
}
```

That traceability is deliberate. A summary without source ids would be
indistinguishable from AARON having invented something, and an assistant that
invents its own memory is worse than one that forgets.

`forget_context()` and `cancel` clear the ring. Neither touches SQLite.

### SQLite settings

| Setting | Why |
|---|---|
| `journal_mode = WAL` | A long read does not block the agent. Costs a `-wal` sidecar that grows between checkpoints — hence `checkpoint()` and `on_disk_bytes()`. |
| `synchronous = NORMAL` | Survives process crashes; only risks the last transactions on power loss. |
| `cache_size = -N` | A **negative** value is a page-count limit, not a byte limit, which is what makes it a real bound. SQLite cannot read past it, so the page cache cannot quietly page the whole database into RAM. |
| `temp_store = FILE` | Sorts and joins spill to disk rather than to heap. |
| `mmap_size = 0` | No memory-mapped I/O, so database pages are accounted for by the page cache above rather than invisible. |

---

## 8. Stress testing and leak detection

`aaron resources stress` runs five workloads and reports a verdict for each:

| Workload | What it exercises |
|---|---|
| intent classification | the feature extractor and tensor arena |
| memory database churn | open, write, query, prune, close |
| allocation churn | whether the working set comes back |
| audio buffer churn | WAV encode of realistic frame sizes |
| idle resource sampling | the monitoring path itself |

### How the verdict is reached

RSS legitimately rises during warm-up as the allocator caches blocks, then
plateaus. Comparing the first sample to the last would call that a leak.

Instead:

1. the first 200 iterations are warm-up and are excluded;
2. the remaining samples are split into thirds;
3. the median of the first third is compared with the median of the last third;
4. the verdict is `growing` only if the later median exceeds the earlier one by
   more than **max(2 MiB, 1% of baseline)**;
5. fewer than 9 post-warm-up samples reports `insufficient_data` rather than
   guessing.

The tolerance is derived from observed noise, not picked to make a failure
disappear. The report also prints a least-squares slope in bytes per thousand
iterations, which is the number that trends upward on a real leak.

### Measured on the development machine

Arch Linux, AMD Ryzen 5 7535HS (12 cores), 22.7 GiB RAM, RTX 2050 4 GB,
release build, `aaron resources stress --iterations 20000`:

```
intent classification            n=20000   peak  8.9 MiB  final  8.9 MiB  slope  13364/1k  stable
memory database churn            n=2000    peak  9.1 MiB  final  9.1 MiB  slope  21588/1k  stable
allocation churn                 n=5000    peak  9.2 MiB  final  9.2 MiB  slope   3934/1k  stable
audio buffer churn               n=5000    peak  9.3 MiB  final  9.3 MiB  slope   3934/1k  stable
idle resource sampling           n=2000    peak  9.3 MiB  final  9.3 MiB  slope      0/1k  stable
```

No workload showed unbounded growth. GPU memory was 39 MiB before and after,
so AARON itself adds nothing to the card on this workload.

The slopes are small positive numbers — roughly 100–400 KiB across the whole
run — which is allocator warming, not growth: the median-of-thirds comparison
that decides the verdict sees a flat line. That is exactly why the verdict uses
a robust statistic and the slope is reported separately rather than the slope
being the pass/fail criterion.

Re-run on your own hardware before trusting these numbers anywhere else.

---

## 9. What is not claimed

* Releasing a model returns AARON's reference. It does **not** promise the OS
  reclaims the pages immediately, and AARON does not clear CUDA caches it does
  not own. Candle's CPU allocator caches freed blocks; `aaron resources status`
  reports measured RSS precisely because the derived figure would be a guess.
* GPU memory is reported for the whole card, not for AARON, because AARON
  cannot separate its own VRAM from the desktop's without a CUDA context it does
  not hold.
* The static footprint (`DERIVED_STATIC_FOOTPRINT`) is labelled derived
  everywhere it appears, and the measured RSS is shown beside it.
---

## 10. Speech engines

The ASR and TTS backends are not loaded at startup. They go through the same
`ModelSlot` as the intent model: loaded on first use, released after
`idle_unload_seconds` (default 120) of no use, and never released while a
caller holds a guard. `[resources.models]` controls this:
`load_asr_on_demand`, `load_tts_on_demand`, `unload_idle_models`,
`idle_unload_seconds`.

* **Phonon (ASR).** AARON does not link the engine; it runs the `fermion`
  command as a child process through the same hardened harness as the tool
  layer (timeout, process group, cleared environment, capped output). The
  engine's resident memory is modest, but torch and the ONNX runtimes reserve
  a large *virtual* address space, so the default 4 GiB `RLIMIT_AS` cap makes
  them fail to allocate a single megabyte. AARON therefore runs the engine
  with `.without_address_space_limit()`; the 120-second timeout is what bounds
  it. The Phonon-2 weights are 164 MB to download and 178 MB unpacked, are
  CC-BY-4.0, and are fetched by the engine on first use into
  `~/.cache/fermion/speech/`. AARON never downloads a model.
* **piper (TTS).** AARON holds the backend (model path, rate, player lookup)
  in the slot and invokes the `piper` binary per utterance, rendering to a
  WAV and handing it to `paplay`, `aplay` or `pw-play`. The voice file (63 MB
  for `en_US-lessac-medium`) is opened by piper itself; AARON does not keep
  it resident between utterances, but the backend instance is kept for the
  session by the slot.

What is already measured, and not re-measured here:

| measurement | value | source |
|---|---|---|
| machine | Arch Linux, AMD Ryzen 5 7535HS, 22.7 GiB RAM, RTX 2050 4 GB | `docs/release-0.5-report.md` §7 |
| AARON inference, intent only | 0.56 ms (200 iterations, three runs) | `docs/release-0.5-report.md` §7 |
| AARON inference, with readiness head | 0.78–0.80 ms | `docs/release-0.5-report.md` §7 |
| parameters, intent model | 1,000,000 (4 MB f32) | `docs/release-0.5-report.md` §7 |
| stress-test RSS, all five workloads | 8.9–9.3 MiB peak, stable | `docs/resources.md` §8 |
| audio buffer at defaults | 640 KiB (10 s, 16 kHz mono f32) | `docs/resources.md` §6 |
| Phonon-2 download / unpacked | 164 MB / 178 MB | task facts |
| Phonon-2 average word error | 5.21 % over seven English benchmarks | task facts |
| piper voice (`en_US-lessac-medium`) | 63 MB | task facts |
