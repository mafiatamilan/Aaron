# Offline and dependency audit — AARON 1.0

## Verdict

AARON's *source* contains no networking code: there is no `TcpStream`, `UdpSocket`, `reqwest`, `ureq`, `hyper`, `tokio`, or `curl`/`wget` invocation anywhere in `crates/`, and `Cargo.lock` contains no HTTP client crate. The offline guarantee is structural and is backed by `crates/aaron-asr/src/lib.rs`'s `every_backend_is_offline_only` test. However, "offline" is not the same as "works on a fresh install": on this machine (Arch, PipeWire-only, no speech packages) a default build cannot capture audio (ALSA `default` is a playback endpoint), all three ASR backends degrade to the `constrained-grammar` demo, none of the four TTS backends can actually play sound (espeak renders to `/dev/null`, piper's `render_wav` always errors, spd-say is not installed), and `aaron doctor` labels the constrained demo as **ready**. The 1.0 user gets a working text assistant out of the box and nothing that speaks or listens.

## The real dependency table

| component | needs | size | network? | present on this machine? | works without it? |
|---|---|---|---|---|---|
| Build toolchain | `rust`/`cargo`, `pkgconf`, C toolchain (`base-devel`/`build-essential`) | n/a | no (distro packages) | yes | no (cannot build) |
| ALSA headers (cpal) | `alsa-lib` dev headers (`libasound2-dev`) | n/a | no | no (`pkg-config` cannot find `libasound`; `/usr/include/alsa` exists) | yes, via `--no-default-features` (text-only binary) |
| Audio capture (PipeWire) | `libpipewire-0.3` + `libspa-0.2` dev headers; running PipeWire | n/a | no | yes (`pkg-config` finds 1.6.9 / 0.2) | capture silently records silence on ALSA `default` |
| ASR: whisper.cpp | `whisper-cli` binary + a ggml model file | binary ~pkg; model 75 MB (tiny) – 3.1 GB (large-v3) | one fetch for the model (and the binary if not packaged) | no (`whisper-cli` absent) | degrades to `constrained-grammar` demo |
| ASR: constrained-grammar | nothing | 0 | no | yes (always) | n/a — it *is* the fallback; emits 1 of 7 fixed phrases, no acoustic analysis |
| ASR: disabled | nothing | 0 | no | yes (opt-in) | text-only / errors |
| TTS: espeak-ng | `espeak-ng` binary | package ~3 MB | no | no | no audio (and its `speak` renders to `/dev/null` anyway) |
| TTS: piper | `piper` binary + ONNX voice | binary 26.5 MB (x86_64 tarball) + ~63 MB per voice | one fetch for binary and voice | no | no (and `render_wav` always errors) |
| TTS: spd-say | `spd-say` binary (`speech-dispatcher`) | package ~1 MB | no | no | no |
| TTS: text-only | nothing | 0 | no | yes (always) | yes (prints replies) |
| Classifier checkpoint | `models/brain/aaron-intent-1m.ckpt` (4.0 MB f32) | 4.0 MB | no (trained locally) | no (only `models/brain/.gitkeep`) | yes (deterministic router) |
| Neural wake word | none shipped | 0 | n/a | n/a | transcript/energy matching only |

A fully working *voice* install (whisper `base.en` 147.96 MB + one piper voice 63.20 MB + checkpoint 4.0 MB + piper binary 26.46 MB) costs **≈ 241.6 MB** of downloads/models on top of the distro packages; with `ggml-tiny.en` (77.70 MB) instead of base it is **≈ 171.4 MB**. Nothing in that set is fetched by AARON.

## What a fresh install can do today

Works, no extra packages (verified by `cargo run --release -p aaron-app -- ...`):

- `aaron setup`, `aaron doctor`, `aaron model`, `aaron benchmark`
- `aaron run`, `aaron ask "<text>"` — text turns end to end (classifier falls back to the deterministic router)
- `aaron tools list`, `aaron permissions show/test`
- `aaron memory list|search|add|delete|backup`
- `aaron learning pending|approve|reject|evaluate|analyse|safety|promote|rollback`
- `aaron brain train` — trains a candidate checkpoint from the shipped `training/dataset/current.jsonl` (verified: completes in ~1.2 s for 1 epoch)
- `aaron dataset check|report`
- `aaron resources status|limits|stress|soak|checkpoint`
- `aaron audio devices` (lists ALSA/PipeWire devices, with harmless probe warnings)
- `aaron logs`

Does not work (or works only partially) on a fresh install:

- `aaron audio test` on a default build: reports "The capture device delivered no audio at all" (verified) — rebuild with `--features aaron-audio/capture-pipewire`.
- `aaron listen`: capture is silent on the default build; with `capture-pipewire` it captures, but ASR is the constrained demo, so it only ever "hears" one of 7 canned phrases and the wake phrase always matches.
- Speech recognition: no. `constrained-grammar` is not recognition.
- Speech synthesis: no audio. `auto` selects `espeak-ng` if present (it is not), else `text-only`; even when espeak is installed it renders to `/dev/null`.
- The 1 M intent model is not deployed (`aaron brain train` writes a *candidate*; promotion is refused by the configured thresholds).

## Offline claim: verified or not

Verified as far as the Rust source goes.

Evidence:
- `grep -RInE "TcpStream|UdpSocket|TcpListener|reqwest|ureq|hyper::|curl|wget|std::net" --include='*.rs'` returns only the self-test in `crates/aaron-asr/src/lib.rs:118-126` and untrusted-content scrubbing in `crates/aaron-agent/src/untrusted.rs:30-31` (those are *patterns stripped from file/process output*, not calls).
- `grep -RIn "Command::new"` shows only fixed program names: `whisper-cli`, `espeak-ng`/`piper`/`spd-say`, and hardware probes (`nvidia-smi`, etc.) in `crates/aaron-core/src/hardware.rs:354` and `crates/aaron-tools/src/proc.rs:175`.
- `Cargo.lock` contains no `reqwest`/`hyper`/`ureq`/`tokio`/`mio`.
- `config/default.toml` sets `offline_only = true` and `security.allow_network = false`; `config/permissions.toml` keeps `allow_network = false` and `open_url` is `ask` + network-gated.

Caveat: "no socket in AARON" is true, but AARON can still *launch* a browser via `open_url` (a user-approved, scheme-checked external process) and `start_application`. Those are gated by policy and confirmation, but they are the only places a user can cause network activity. AARON itself never initiates it.

There is also **no model download path at all**: `WhisperAsr::new` returns `Error::missing_with_hint` when `model_path` is empty (`crates/aaron-asr/src/whisper.rs:39-47`), and `PiperTts` reports "AARON never downloads models" (`crates/aaron-tts/src/lib.rs:193`). `scripts/install-arch.sh` explicitly refuses to fetch models.

## Defects and false claims

| id | severity | what | file:line | how to confirm | suggested fix |
|---|---|---|---|---|---|
| D1 | high | `aaron doctor` prints `constrained-grammar — ready` for a backend that is explicitly *not* speech recognition (`ready: true` in `ConstrainedAsr::info`). The problems list skips it because of `ready`, so a fresh install looks complete. | `crates/aaron-asr/src/constrained.rs:52`; `crates/aaron-app/src/commands.rs:149`, `:233` | `cargo run --release -p aaron-app -- doctor` → "speech recog. constrained-grammar — ready" | Make `ready` false for the constrained backend, or add a separate `demo`/`not-speech-recognition` status that doctor and the problems list treat as not-ready |
| D2 | high | `EspeakTts::speak` renders to `/dev/null` and discards the boolean (`render_wav(...).map(|_| ())`), so a reply that was never audible is reported as spoken (`Ok(())`). | `crates/aaron-tts/src/lib.rs:79-80` | inspect; with espeak installed, `speak` returns `Ok` while producing no sound | `speak` should invoke espeak without `-w` (playback) or report failure; stop discarding the rendered flag |
| D3 | high | `PiperTts::render_wav` unconditionally returns `Err`, yet `auto` prefers piper over `spd-say`; a machine with a piper model gets no audio and the turn is reported as text | `crates/aaron-tts/src/lib.rs:182-196`, `:358-372` | build with a piper model present; `speak` errors and `speak_or_echo` prints instead of speaking | either implement piper playback (pipe stdout to a player) or demote piper below `spd-say`/`espeak-ng` in `auto` |
| D4 | high | `auto` order is piper → espeak-ng → spd-say, and espeak renders to `/dev/null` (D2), so on any machine with espeak installed AARON believes it spoke while producing silence | `crates/aaron-tts/src/lib.rs:358-372` | same as D2 | fix D2, or reorder so a genuinely playing backend is preferred |
| D5 | medium | `asr.use_gpu` is inverted: code adds whisper.cpp's `-ng` ("no GPU") flag when `use_gpu == true`, disabling the GPU exactly when the user asks for it | `crates/aaron-asr/src/whisper.rs:88-91` | `cfg.use_gpu = true` → argv contains `-ng` (test at `:314-322` only checks presence) | add `-ng` when `!use_gpu` (or remove the flag and document whisper.cpp's default) |
| D6 | medium | Default audio capture uses CPAL's ALSA host; on this PipeWire machine `default` is a playback endpoint, so capture records silence. Fixed only by rebuilding with `capture-pipewire`. README/install does not make this the default. | `crates/aaron-audio/Cargo.toml:14-24`; `crates/aaron-app/src/commands.rs:459-475`; verified `cargo run ... -- audio test` → "capture device delivered no audio" | run `aaron audio test` on a default build | make `capture-pipewire` the default on Linux, or auto-detect PipeWire and select the host |
| D7 | medium | No repeated-trigger protection in the wake matcher: `TranscriptMatcher::check` increments `activations` on every call with no debounce/dedup, and `listen_once` strips only the leading wake phrase, so "hey aaron hey aaron do X" leaves a second wake phrase in the dispatched command | `crates/aaron-wake/src/lib.rs:160-168`; `crates/aaron-agent/src/session.rs:105-112`, `:159-178` | unit test with a double wake phrase | strip all leading wake phrases / add a per-session debounce window |
| D8 | low | `aaron doctor` reports `audio source` with the playback device first on this machine (ALSA probe noise), which is cosmetic but confusing | `crates/aaron-app/src/commands.rs:160-180` (verified output) | run doctor | filter capture-capable devices / label endpoints |
| D9 | low | The task's "containment bonus scores 'they said hey aaron was here' at 1.0000" describes *removed* code: current `phrase_similarity` scores that utterance **0.4444** (verified via a small harness). The comment documents the old behaviour. The finding is now stale, but the comment should be reconciled with tests so the history is not misleading. | `crates/aaron-wake/src/lib.rs:56-68` | harness: `phrase_similarity("hey aaron", "they said hey aaron was here")` → 0.4444 | update the comment/test expectations to the current algorithm |
| D10 | low | `TextOnlyTts::speak` returns `Ok(())` and `speak_or_echo` prints the reply only when `echo` is true; `aaron ask` sets `echo_to_terminal = false`, so a text-only install still gets the reply from `ask`'s own `println!` — works, but the "spoken" status is implicit | `crates/aaron-tts/src/lib.rs:314-320`, `:335-348`; `crates/aaron-app/src/commands.rs:138` | `aaron ask "hello"` on a text-only build | make delivery return whether audio was produced |

## Missing tests that matter for 1.0

- TTS: assert `EspeakTts::speak` either plays audio or returns an error; assert the rendered flag is not discarded; assert `auto` never selects a backend whose `speak` cannot produce audio.
- TTS: piper availability vs. speakability (currently conflated) — a test that a selected backend's `speak` succeeds.
- ASR: doctor-level test asserting the constrained backend is not labelled "ready".
- Wake: test that a wake phrase spoken twice in one utterance does not re-trigger; test `phrase_similarity("hey aaron", "they said hey aaron was here")` against the intended threshold.
- Audio: capture smoke test on a PipeWire host (or at least a documented expected level).
- Offline: a workspace-level test that no crate depends on a networking crate (the asr crate has one; promote it to the workspace).
- GPU: test that `use_gpu = true` does *not* emit `-ng`.

## The exact steps to make voice work on Arch, Debian/Ubuntu and Fedora

### Arch Linux

```bash
sudo pacman -S --needed rust pkgconf base-devel alsa-lib alsa-utils \
    pipewire-pulse espeak-ng speech-dispatcher libpipewire libspa
# whisper.cpp + a model:
sudo pacman -S --needed whisper-cpp
# download a model yourself (example: base.en, 147.96 MB):
curl -L -o ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
# optional neural TTS:
# download piper_linux_x86_64.tar.gz (26.46 MB) from
#   https://github.com/rhasspy/piper/releases/latest
# and a voice (example: 63.20 MB) from
#   https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
```

Then configure (`~/.config/aaron/config.toml`):

```toml
[audio]
backend = "cpal"          # rebuilt with capture-pipewire, see below

[asr]
backend = "auto"
model_path = "/path/to/ggml-base.en.bin"

[tts]
backend = "auto"          # or "spd-say"/"espeak-ng"
model_path = "/path/to/en_US-lessac-medium.onnx"   # only if using piper
```

Build with PipeWire capture:

```bash
cargo build --release --features aaron-audio/capture-pipewire
sudo install -Dm755 target/release/aaron /usr/local/bin/aaron
aaron doctor
aaron listen
```

### Debian / Ubuntu

```bash
sudo apt install build-essential pkg-config libasound2-dev alsa-utils \
    pipewire-pulse espeak-ng speech-dispatcher libpipewire-0.3-dev libspa-0.2-dev
# whisper.cpp (sid/trixie+):
sudo apt install whisper.cpp-tools        # provides /usr/bin/whisper-cli
# on older releases, build whisper.cpp from source
# piper TTS is NOT packaged: fetch the tarball + a voice as above
```

Then the same config and `cargo build --release --features aaron-audio/capture-pipewire`.

### Fedora

```bash
sudo dnf install rust cargo pkgconf-pkg-config alsa-lib-devel alsa-utils \
    pipewire-pulseaudio espeak-ng speech-dispatcher pipewire-devel
sudo dnf install whisper-cpp                 # provides whisper-cli
# piper TTS is NOT packaged: fetch the tarball + a voice as above
```

Then the same config and `cargo build --release --features aaron-audio/capture-pipewire`.

### Model download URLs and sizes (verified via `curl -sIL`)

| model | URL | size |
|---|---|---|
| `ggml-tiny.en` | https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin | 77,704,715 B (74.1 MiB) |
| `ggml-base.en` | https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin | 147,964,211 B (141.1 MiB) |
| `ggml-small.en` | https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin | 487,614,201 B (465.0 MiB) |
| `ggml-medium.en` | https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin | 1,533,774,781 B (1.43 GiB) |
| `ggml-large-v3` | https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin | 3,095,033,483 B (2.88 GiB) |
| piper (x86_64) | https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz | 26,460,462 B (25.2 MiB) |
| piper voice (lessac medium) | https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx | 63,201,294 B (60.3 MiB) |
| classifier checkpoint | produced by `aaron brain train` | 4,000,000 B (3.8 MiB) |

Only the model/binary/voice fetches above need a network, and they are performed by you, never by AARON. AARON itself opens no socket.
