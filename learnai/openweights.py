"""Run released open-weight language models locally on Apple silicon.

The rest of ``learnai`` rebuilds models small enough to read. This module is the
other end of the telescope: download a real checkpoint — billions of trained
parameters — and run it on this laptop, so the pieces from the lessons (the
tokenizer, the chat template, sampling, the KV cache) can be seen at full scale.

Why MLX here and not PyTorch?  Apple silicon has *unified memory*: the GPU reads
the same RAM as the CPU, so a 16 GB checkpoint needs 16 GB of RAM rather than a
16 GB graphics card.  MLX is Apple's array framework built around that, and the
``mlx-community`` organisation on Hugging Face republishes most open-weight
models already quantised to 4 bits — roughly half a byte per parameter, plus a
little for the per-group scales:

    bytes ≈ parameters × 4.5 / 8          (4-bit weights, one scale per 64 of them)

Two numbers decide whether a model is pleasant to use, and :func:`chat` prints
both after every reply so you can measure rather than guess:

* **peak memory** — must stay under what macOS lets the GPU keep resident
  (:func:`gpu_budget_gb`), or the machine starts swapping;
* **generation tokens/s** — decoding is memory-bandwidth bound, so it scales with
  the bytes read *per token*.  A mixture-of-experts model reads only its active
  experts, which is why ``qwen3.6-35b-a3b`` (35B total, ~3B active) decodes far
  faster than the dense ``qwen3.8-27b`` despite being larger on disk.

Python::

    from learnai.openweights import load, generate
    model, tokenizer = load("qwen3.5-4b")
    print(generate(model, tokenizer, "用一句话解释注意力机制。"))

Shell::

    make models                     # what fits this Mac, what is already downloaded
    make pull MODEL=qwen3.8-27b     # download once (into `models_dir()`)
    make chat MODEL=qwen3.8-27b     # interactive chat; /think, /reset, /exit

Where the weights are stored does not change how fast a model generates: they
are read into memory once, by :func:`load`, and every token after that is
computed from RAM.  The disk only sets the load time — a USB hard disk reads
about 120 MB/s, an internal SSD about 3 GB/s, so a 16 GB model starts in ~2
minutes from the first and ~5 seconds from the second.  ``LEARNAI_MODELS_DIR``
moves the store, e.g. to a large external drive (see :func:`models_dir`).
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from collections.abc import Iterator
from pathlib import Path

DEFAULT_MODEL = "qwen3.5-4b"

# Files an MLX checkpoint actually needs. Skipping the rest (README images, GGUF
# copies some repos carry) keeps downloads to what `mlx_lm.load` reads.
_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.py", "*.jinja", "*.txt", "*.model", "*.tiktoken"]


@dataclasses.dataclass(frozen=True)
class ModelCard:
    """One downloadable checkpoint and what it costs to run."""

    key: str
    repo: str  # Hugging Face repo id, "org/name"
    download_gb: float  # summed from the Hub's file listing (measured 2026-11-06)
    kind: str  # "dense" — every weight is read per token; "moe" — only the routed experts
    note: str
    # Set when the Hub release is full precision (bf16) and nobody has published an
    # MLX quantisation: the weights are quantised to this many bits here, once, on
    # first use (see `pull`).  The download is then ~3.5x larger than what is run.
    quantize: int | None = None

    @property
    def run_gb(self) -> float:
        """Size of the weights as they are loaded — the number that memory cares about."""
        if self.quantize is None:
            return self.download_gb
        return self.download_gb * (self.quantize + 0.5) / 16  # bf16 -> n-bit + a scale per 64


# Curated for a 32 GB Apple-silicon Mac: every entry loads, and the list spans the
# trade-offs worth feeling — size vs quality, dense vs mixture-of-experts, and two
# model families.  Any other "org/name" repo id works too; see `resolve`.
MODELS: dict[str, ModelCard] = {
    card.key: card
    for card in (
        ModelCard(
            "qwen3-0.6b",
            "mlx-community/Qwen3-0.6B-4bit",
            0.35,
            "dense",
            "tiny — smoke tests and reading internals, not for real questions",
        ),
        ModelCard(
            "qwen3.5-4b",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            3.06,
            "dense",
            "default — fast everyday model, strong Chinese and English",
        ),
        ModelCard(
            "qwen3.5-9b",
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            5.98,
            "dense",
            "noticeably smarter than 4B, still quick",
        ),
        ModelCard(
            "gemma4-e4b",
            "mlx-community/gemma-4-e4b-it-4bit",
            5.18,
            "dense",
            "Google's small Gemma 4 — a second family to compare against Qwen",
        ),
        ModelCard(
            "gemma4-12b",
            "mlx-community/gemma-4-12B-it-4bit",
            6.74,
            "dense",
            "Google's mid-size Gemma 4; reasons only when asked (/think)",
        ),
        ModelCard(
            "dianjin-r1-7b",
            "DianJin/DianJin-R1-7B",
            15.23,
            "dense",
            "Alibaba Cloud's finance reasoning model on a Qwen2.5-7B base; quantised here",
            quantize=4,
        ),
        ModelCard(
            "gpt-oss-20b",
            "mlx-community/gpt-oss-20b-MXFP4-Q8",
            12.10,
            "moe",
            "OpenAI's open-weight MoE; always reasons before answering",
        ),
        ModelCard(
            "gemma4-26b-a4b",
            "mlx-community/gemma-4-26b-a4b-it-4bit",
            15.37,
            "moe",
            "26B total, ~4B active — big-model quality at small-model speed",
        ),
        ModelCard(
            "qwen3.8-27b",
            "mlx-community/Qwen3.8-27B-4bit",
            16.08,
            "dense",
            "the strongest dense model that fits comfortably; slowest per token",
        ),
        ModelCard(
            "qwen3.6-35b-a3b",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
            20.43,
            "moe",
            "35B total, ~3B active — the largest here; close other heavy apps first",
        ),
    )
}


def resolve(name: str | None = None) -> ModelCard:
    """A curated key, ``None`` for the default, or any Hugging Face ``org/name`` repo id."""
    name = name or DEFAULT_MODEL
    if name in MODELS:
        return MODELS[name]
    if name.count("/") == 1:
        # Uncurated: size unknown until downloaded, so `fit` reports "unknown".
        return ModelCard(name, name, math.nan, "?", "not curated — any MLX repo id works")
    raise KeyError(
        f"unknown model {name!r}; choose one of: {', '.join(MODELS)} "
        "(or pass a Hugging Face repo id such as mlx-community/Qwen3-8B-4bit)"
    )


# --------------------------------------------------------------------------------------
# Memory: will it fit?
# --------------------------------------------------------------------------------------


def physical_memory_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def gpu_budget_gb() -> float:
    """How much memory Metal recommends the GPU keep resident on this machine.

    It is below total RAM — macOS reserves the rest for itself and other apps.
    MLX reports the real figure (a 32 GiB M6 reports 26.8 GB, ~78%); without MLX
    we fall back to a conservative two thirds of RAM.
    """
    try:
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return info["max_recommended_working_set_size"] / 1e9
    except Exception:
        return physical_memory_gb() * 2 / 3


def fit(download_gb: float, budget_gb: float, overhead_gb: float = 1.5) -> str:
    """Will a checkpoint of ``download_gb`` run inside ``budget_gb``?

    Resident memory ≈ the weights + a KV cache that grows with the conversation
    + the runtime's own buffers.  ``overhead_gb`` covers the last two for a few
    thousand tokens of context.  Returns ``"ok"``, ``"tight"`` (the weights fit
    but long conversations may push it into swap), ``"too big"`` or ``"unknown"``.
    """
    if math.isnan(download_gb):
        return "unknown"
    if download_gb + overhead_gb <= budget_gb:
        return "ok"
    if download_gb <= budget_gb:
        return "tight"
    return "too big"


# --------------------------------------------------------------------------------------
# Download: the Hugging Face cache (shared by every project on this machine)
# --------------------------------------------------------------------------------------


def _offline() -> bool:
    return os.environ.get("LEARNAI_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


class DriveNotConnected(RuntimeError):
    """``LEARNAI_MODELS_DIR`` points at an external volume that is not mounted."""


def models_dir() -> Path | None:
    """Where checkpoints are stored: ``$LEARNAI_MODELS_DIR``, else ``None`` (the default
    Hugging Face cache, ``~/.cache/huggingface/hub``).

    The layout inside is the Hugging Face cache layout either way, so a directory
    can be moved between disks and pointed at again.  For a directory under
    ``/Volumes/<drive>`` this checks that the drive is plugged in, so a missing
    drive is reported as such — never mistaken for "not downloaded", which would
    start a multi-gigabyte download somewhere else.
    """
    raw = os.environ.get("LEARNAI_MODELS_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.parts[:2] == ("/", "Volumes") and len(path.parts) > 2:
        volume = Path(*path.parts[:3])
        if not volume.is_mount():
            raise DriveNotConnected(f"models are stored on {volume}, which is not connected")
    return path


def cached_sizes() -> dict[str, float]:
    """``repo id -> GB`` already downloaded into :func:`models_dir`.  Never touches the network."""
    cache_dir = models_dir()  # raises DriveNotConnected: an unplugged drive is not "empty"
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(cache_dir)
    except Exception:  # not installed, or nothing downloaded yet
        return {}
    return {r.repo_id: r.size_on_disk / 1e9 for r in info.repos if r.repo_type == "model"}


def quantized_path(card: ModelCard) -> Path:
    """Where a model quantised here is kept: beside the Hub cache, named after its source."""
    from huggingface_hub import constants

    base = models_dir() or Path(constants.HF_HUB_CACHE)
    return base / "quantized" / f"{card.repo.replace('/', '--')}-{card.quantize}bit"


def local_path(name: str | None = None) -> Path | None:
    """The downloaded snapshot directory for a model, or ``None`` if it was never downloaded."""
    from huggingface_hub import snapshot_download

    card = resolve(name)
    if card.quantize is not None:
        path = quantized_path(card)
        return path if (path / "config.json").exists() else None
    try:
        path = snapshot_download(
            card.repo,
            allow_patterns=_ALLOW_PATTERNS,
            cache_dir=models_dir(),
            local_files_only=True,
        )
    except Exception:  # LocalEntryNotFoundError, and its older spellings
        return None
    return Path(path)


def pull(name: str | None = None) -> Path:
    """Download a model into :func:`models_dir` (resumable; a no-op if complete).

    A card with ``quantize`` set is downloaded at full precision and then quantised
    once, with ``mlx_lm.convert`` — the same affine 4-bit, group-64 scheme the
    ``mlx-community`` checkpoints use, so the result is comparable with them.
    Converting needs the bf16 weights in memory (~2 bytes per parameter).
    """
    if _offline():
        raise RuntimeError("LEARNAI_OFFLINE is set; unset it to download models")
    from huggingface_hub import snapshot_download

    card = resolve(name)
    source = Path(
        snapshot_download(card.repo, allow_patterns=_ALLOW_PATTERNS, cache_dir=models_dir())
    )
    if card.quantize is None:
        return source
    target = quantized_path(card)
    if not (target / "config.json").exists():
        _quantize(source, target, card.quantize)
    return target


def _quantize(source: Path, target: Path, bits: int) -> None:
    """``mlx_lm.convert`` from a copy of the checkpoint on the internal disk.

    The converter maps the safetensors files into memory and lets the GPU read the
    weights straight out of them.  On a USB hard disk that read is a chain of page
    faults at ~130 MB/s inside a GPU command, which Metal aborts as a timeout after
    a few seconds.  Copying the files to the (fast) internal disk first costs one
    sequential read of the checkpoint and makes the conversion routine.
    """
    import shutil
    import tempfile

    from mlx_lm.convert import convert

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(target, ignore_errors=True)  # a half-written attempt would be refused
    with tempfile.TemporaryDirectory(prefix="learnai-convert-") as tmp:
        staged = Path(tmp) / source.name
        shutil.copytree(source, staged, symlinks=False)  # resolve the cache's blob links
        convert(str(staged), str(target), quantize=True, q_bits=bits, q_group_size=64)


def load(name: str | None = None):
    """``(model, tokenizer)`` for a model, downloading it first if it is not cached.

    Loading from the local snapshot path rather than the repo id means a cached
    model starts without any network round-trip, and works offline.
    """
    from mlx_lm import load as mlx_load

    path = local_path(name)
    if path is None:
        if _offline():
            raise RuntimeError(f"{resolve(name).key} is not downloaded; run `make pull MODEL=...`")
        path = pull(name)
    return mlx_load(str(path))


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def chat_prompt(tokenizer, messages: list[dict[str, str]], think: bool = False) -> str:
    """Render a conversation into the exact text the model was trained on.

    Every model family wraps turns in its own special tokens; the template ships
    with the checkpoint.  ``enable_thinking`` is read by Qwen and Gemma templates
    (Qwen reasons by default, Gemma only when asked) and ignored by the rest.
    """
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=think
    )


def stream(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    think: bool = False,
    seed: int | None = None,
) -> Iterator:
    """Yield ``mlx_lm`` responses piece by piece; each carries ``.text`` and live statistics.

    ``temperature=0`` is greedy decoding — deterministic, handy for tests.  The
    whole conversation is re-encoded every turn; that keeps the code simple, and
    prefill (reading the prompt) runs far faster than generation anyway.
    """
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    if seed is not None:
        mx.random.seed(seed)
    prompt = chat_prompt(tokenizer, messages, think=think)
    sampler = make_sampler(temp=temperature, top_p=top_p if temperature > 0 else 0.0)
    yield from stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler)


def generate(model, tokenizer, prompt: str, *, system: str | None = None, **kwargs) -> str:
    """One question, one answer.  Keyword arguments are those of :func:`stream`."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return "".join(piece.text for piece in stream(model, tokenizer, messages, **kwargs))


def _stats(last) -> str:
    return (
        f"[{last.generation_tokens} tokens · {last.generation_tps:.1f} tok/s · "
        f"prompt {last.prompt_tokens} tok at {last.prompt_tps:.0f} tok/s · "
        f"peak {last.peak_memory:.1f} GB]"
    )


def chat(
    name: str | None = None,
    *,
    prompt: str | None = None,
    system: str | None = None,
    think: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    patch: str | None = None,
) -> int:
    """Interactive chat in the terminal, or a single answer when ``prompt`` is given.

    ``patch`` is a weight edit written by ``tools/weight_edit_experiment.py edit``;
    it is applied on top of the stored weights after loading (see ``learnai.weightedit``).
    """
    card = resolve(name)
    budget = gpu_budget_gb()
    verdict = fit(card.run_gb, budget)
    print(f"model   {card.key}  ({card.repo})")
    print(f"memory  {card.run_gb:.1f} GB weights vs {budget:.1f} GB GPU budget -> {verdict}")
    if verdict == "too big":
        print("        this will swap heavily; pick a smaller model (`make models`)")

    source = local_path(card.key)
    print(f"from    {source or 'not downloaded yet: downloading first'}")
    import mlx_lm  # noqa: F401 — import before the clock starts: time the disk, not Python

    started = time.perf_counter()
    model, tokenizer = load(card.key)
    seconds = time.perf_counter() - started
    # Disk -> memory throughput: what the storage location actually costs you.
    rate = f" ({card.run_gb / seconds * 1000:.0f} MB/s)" if source and card.run_gb else ""
    print(f"loaded  in {seconds:.1f}s{rate}")
    if patch:
        from learnai.weightedit import apply_patch, load_patch

        edits, _ = load_patch(patch)
        apply_patch(model, edits)
        changed = sum(e.n_weights for e in edits)
        print(f"patch   {patch}: {changed:,} weights in blocks {[e.layer for e in edits]}")
    print()

    messages: list[dict[str, str]] = [{"role": "system", "content": system}] if system else []

    def reply() -> None:
        pieces, last = [], None
        for last in stream(
            model,
            tokenizer,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            think=think,
        ):
            pieces.append(last.text)
            print(last.text, end="", flush=True)
        messages.append({"role": "assistant", "content": "".join(pieces)})
        if last is not None:
            print(f"\n\033[2m{_stats(last)}\033[0m")

    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
        reply()
        return 0

    print("commands: /think (toggle reasoning)  /reset (forget history)  /exit  — or Ctrl-D\n")
    while True:
        try:
            line = input("\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            return 0
        if line == "/reset":
            del messages[1 if system else 0 :]
            print("(history cleared)")
            continue
        if line == "/think":
            think = not think
            print(f"(reasoning {'on' if think else 'off'})")
            continue
        messages.append({"role": "user", "content": line})
        print("\033[1mmodel>\033[0m ", end="", flush=True)
        try:
            reply()
        except KeyboardInterrupt:  # stop a runaway answer, keep the session
            messages.pop()
            print("\n(interrupted)")
