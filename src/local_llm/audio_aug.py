"""Audio augmentation for robustness tests: background noise and telephone-quality audio.

Everything is generated locally with numpy/scipy; nothing is downloaded. All functions take
and return mono float32 samples in [-1, 1] at a known sample rate (16 kHz by default), and
every random choice comes from a ``seed`` so the same call gives the same audio.

    samples = load("talk.wav")                      # any format ffmpeg reads -> 16 kHz mono
    noisy = add_noise(samples, 5, kind="babble", babble_sources=["a.wav", "b.wav"])
    phone = telephone(samples)                     # 300-3400 Hz, 8 kHz, mu-law
    save("talk_babble5.wav", noisy)

Noise kinds:

- ``white``: flat-spectrum Gaussian noise.
- ``pink``: 1/f noise (more energy in the low frequencies, like fans and traffic).
- ``babble``: several talkers mixed at random offsets. Other speech files can be given as
  ``babble_sources`` (paths or arrays); without them the input itself is reversed and shifted,
  which keeps a speech-like spectrum without intelligible words.
- ``office``: pink noise plus soft babble plus occasional keyboard-like clicks.

``add_noise`` scales the noise so the signal-to-noise ratio over the whole clip is exactly
``snr_db``; if the mix would clip, the whole mix is scaled down, which keeps the ratio.
"""

import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SR = 16000
NOISE_KINDS = ("white", "pink", "babble", "office")
BABBLE_TALKERS = 6


def load(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any audio file ffmpeg can read into mono float32 samples at ``sr``."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<f4").astype(np.float32)


def save(path: str | Path, samples: np.ndarray, sr: int = SR) -> Path:
    """Write mono samples as a 16-bit PCM WAV file (values outside [-1, 1] are clipped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0) * 32767).round()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.astype("<i2").tobytes())
    return path


def power(samples: np.ndarray) -> float:
    """Mean power of a signal."""
    x = np.asarray(samples, dtype=np.float64)
    return float(np.mean(x * x)) if x.size else 0.0


def snr(clean: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio in dB between a clean signal and the noise added to it."""
    return 10 * np.log10(power(clean) / max(power(noise), 1e-20))


def _unit(x: np.ndarray) -> np.ndarray:
    p = power(x)
    return (x / np.sqrt(p)) if p > 0 else x


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise, shaped in the frequency domain (amplitude ~ 1/sqrt(f))."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spectrum.size, dtype=np.float64)
    f[0] = 1.0
    return _unit(np.fft.irfft(spectrum / np.sqrt(f), n))


def _tile(x: np.ndarray, n: int, offset: int) -> np.ndarray:
    """``n`` samples of ``x`` looped, starting at ``offset``."""
    reps = int(np.ceil((n + offset) / max(len(x), 1))) + 1
    return np.tile(x, reps)[offset : offset + n]


def babble_noise(
    n: int,
    rng: np.random.Generator,
    sources: Sequence[np.ndarray],
    talkers: int = BABBLE_TALKERS,
) -> np.ndarray:
    """Several talkers at random offsets and levels (within 6 dB), summed."""
    sources = [np.asarray(s, dtype=np.float64) for s in sources if len(s) > 0]
    if not sources:
        raise ValueError("babble needs at least one non-empty speech source")
    out = np.zeros(n)
    for k in range(talkers):
        src = sources[k % len(sources)]
        offset = int(rng.integers(0, len(src)))
        gain = 10 ** (rng.uniform(-6, 0) / 20)
        out += gain * _unit(_tile(src, n, offset))
    return _unit(out)


def click_noise(n: int, rng: np.random.Generator, sr: int = SR, rate: float = 2.5) -> np.ndarray:
    """Sparse keyboard-like clicks: short, bright, decaying bursts (``rate`` per second)."""
    out = np.zeros(n)
    count = rng.poisson(rate * n / sr)
    for _ in range(count):
        length = int(sr * rng.uniform(0.004, 0.012))
        start = int(rng.integers(0, max(n - length, 1)))
        burst = np.diff(rng.standard_normal(length + 1))  # differentiate: high-pass
        burst *= np.exp(-np.linspace(0, 6, length)) * rng.uniform(0.4, 1.0)
        out[start : start + length] += burst[: n - start]
    return out


def _babble_sources(
    samples: np.ndarray, babble_sources: Sequence | None, sr: int
) -> list[np.ndarray]:
    if babble_sources:
        return [
            load(s, sr) if isinstance(s, (str, Path)) else np.asarray(s, dtype=np.float64)
            for s in babble_sources
        ]
    # No other speech given: the input reversed (unintelligible, same spectrum), in two halves.
    rev = np.asarray(samples, dtype=np.float64)[::-1]
    half = len(rev) // 2
    return [rev, rev[half:], rev[:half]] if half else [rev]


def make_noise(
    samples: np.ndarray,
    kind: str = "white",
    sr: int = SR,
    seed: int = 0,
    babble_sources: Sequence | None = None,
) -> np.ndarray:
    """Unit-power noise of the given kind, as long as ``samples``."""
    if kind not in NOISE_KINDS:
        raise ValueError(f"unknown noise kind {kind!r}; choose from {NOISE_KINDS}")
    n = len(samples)
    rng = np.random.default_rng(seed)
    if kind == "white":
        return _unit(white_noise(n, rng))
    if kind == "pink":
        return pink_noise(n, rng)
    sources = _babble_sources(samples, babble_sources, sr)
    if kind == "babble":
        return babble_noise(n, rng, sources)
    # office: pink hum, soft distant talkers (-6 dB) and keyboard clicks.
    office = pink_noise(n, rng) + 0.5 * babble_noise(n, rng, sources, talkers=3)
    clicks = click_noise(n, rng, sr)
    if clicks.any():  # peaks about 3x the background level, like a nearby keyboard
        office = office + 3 * np.sqrt(power(office)) * clicks / np.max(np.abs(clicks))
    return _unit(office)


def add_noise(
    samples: np.ndarray,
    snr_db: float,
    kind: str = "white",
    sr: int = SR,
    seed: int = 0,
    babble_sources: Sequence | None = None,
) -> np.ndarray:
    """Mix noise of ``kind`` into ``samples`` at ``snr_db`` dB SNR (over the whole clip).

    ``babble_sources`` (paths or arrays at ``sr``) feed the babble and office kinds. If the
    mix would clip, signal and noise are scaled down together so the SNR is unchanged.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0 or power(x) == 0:
        return x.astype(np.float32)
    noise = make_noise(x, kind, sr, seed, babble_sources)
    noise *= np.sqrt(power(x) / 10 ** (snr_db / 10))
    mix = x + noise
    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix.astype(np.float32)


def mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711-style mu-law companding: compress, quantise to 8 bits, expand."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1  # 256 levels
    return np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu


def telephone(samples: np.ndarray, sr: int = SR) -> np.ndarray:
    """Narrow-band telephone audio: 300-3400 Hz band-pass, 8 kHz sampling, mu-law codec.

    The result is returned at the input rate ``sr`` so it can be fed to the same models.
    """
    from math import gcd

    from scipy.signal import butter, resample_poly, sosfiltfilt

    x = np.asarray(samples, dtype=np.float64)
    if x.size < 32:
        return x.astype(np.float32)
    sos = butter(6, [300, 3400], btype="bandpass", fs=sr, output="sos")
    x = sosfiltfilt(sos, x)
    if sr != 8000:
        g = gcd(sr, 8000)
        x = resample_poly(x, 8000 // g, sr // g)
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x = x / peak
    x = mu_law(x)
    # Phone lines also cut what the codec adds below 300 Hz / above 3400 Hz.
    x = sosfiltfilt(butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos"), x)
    if sr != 8000:
        x = resample_poly(x, sr // g, 8000 // g)
    x = np.pad(x, (0, max(len(samples) - len(x), 0)))[: len(samples)]
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def band_energy(samples: np.ndarray, lo: float, hi: float, sr: int = SR) -> float:
    """Share of the signal's energy between ``lo`` and ``hi`` Hz (0..1)."""
    spec = np.abs(np.fft.rfft(np.asarray(samples, dtype=np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / sr)
    total = float(spec.sum())
    return float(spec[(freqs >= lo) & (freqs < hi)].sum() / total) if total else 0.0


def join(clips: Sequence[np.ndarray], gap_s: float = 0.8, sr: int = SR) -> np.ndarray:
    """Clips one after another with ``gap_s`` seconds of silence between them (none after
    the last), e.g. separate test sentences joined into one longer recording."""
    gap = np.zeros(int(round(gap_s * sr)), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, clip in enumerate(clips):
        if i:
            parts.append(gap)
        parts.append(np.asarray(clip, dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def save_flac(path: str | Path, samples: np.ndarray, sr: int = SR) -> Path:
    """Write mono samples as 16-bit FLAC (lossless, about half the size of a WAV)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0).astype("<f4").tobytes()
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1"]
    cmd += ["-i", "-", "-c:a", "flac", "-sample_fmt", "s16", str(path)]
    subprocess.run(cmd, input=pcm, check=True)
    return path
