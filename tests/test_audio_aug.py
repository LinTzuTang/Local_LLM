"""Audio augmentation: SNR accuracy, determinism, telephone band limits, WAV round trip."""

import numpy as np
import pytest

from local_llm import audio_aug

SR = 16000


def speechy(seconds: float = 3.0, seed: int = 1) -> np.ndarray:
    """A speech-like test signal: a few harmonics with a syllable-rate envelope and pauses."""
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((180, 360, 900, 2400), 1))
    envelope = np.clip(np.sin(2 * np.pi * 3 * t), 0, None)
    rng = np.random.default_rng(seed)
    return (0.1 * tone * envelope + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


def undo_clip_guard(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """The noise that was added, after undoing add_noise's shared anti-clipping gain."""
    gain = float(np.dot(noisy, clean) / np.dot(clean, clean))  # noise is ~uncorrelated
    return noisy.astype(np.float64) / (gain if np.max(np.abs(noisy)) >= 0.989 else 1.0) - clean


@pytest.mark.parametrize("kind", audio_aug.NOISE_KINDS)
@pytest.mark.parametrize("snr_db", [0, 5, 10, 20])
def test_snr_is_achieved(kind, snr_db):
    clean = speechy()
    noisy = audio_aug.add_noise(clean, snr_db, kind=kind, seed=3)
    assert noisy.dtype == np.float32 and len(noisy) == len(clean)
    assert abs(audio_aug.snr(clean, undo_clip_guard(clean, noisy)) - snr_db) <= 0.5


def test_snr_holds_when_the_mix_is_scaled_to_avoid_clipping():
    clean = speechy() * 9  # peaks near 0.9: the 0 dB mix clips and is scaled down
    noisy = audio_aug.add_noise(clean, 0, kind="white", seed=0)
    assert np.max(np.abs(noisy)) <= 0.99 + 1e-6
    noise = audio_aug.make_noise(clean, "white", seed=0) * np.sqrt(audio_aug.power(clean))
    k = float(np.max(np.abs(clean + noise))) / 0.99
    assert k > 1
    assert abs(audio_aug.snr(clean, noisy * k - clean)) <= 0.5


def test_babble_sources_as_arrays_and_paths(tmp_path):
    clean = speechy()
    other = speechy(2.0, seed=7)[::-1].copy()
    path = audio_aug.save(tmp_path / "other.wav", other)
    a = audio_aug.add_noise(clean, 5, kind="babble", babble_sources=[other], seed=2)
    b = audio_aug.add_noise(clean, 5, kind="babble", babble_sources=[path], seed=2)
    assert abs(audio_aug.snr(clean, a - clean) - 5) <= 0.5
    assert abs(audio_aug.snr(clean, b - clean) - 5) <= 0.5


@pytest.mark.parametrize("kind", audio_aug.NOISE_KINDS)
def test_outputs_are_deterministic(kind):
    clean = speechy()
    a = audio_aug.add_noise(clean, 5, kind=kind, seed=11)
    b = audio_aug.add_noise(clean, 5, kind=kind, seed=11)
    c = audio_aug.add_noise(clean, 5, kind=kind, seed=12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.array_equal(audio_aug.telephone(clean), audio_aug.telephone(clean))


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        audio_aug.add_noise(speechy(), 5, kind="rain")


def test_pink_noise_has_more_low_frequency_energy():
    noise = audio_aug.make_noise(np.zeros(SR * 2), "pink", seed=0)
    assert audio_aug.band_energy(noise, 0, 1000) > 0.6
    white = audio_aug.make_noise(np.zeros(SR * 2), "white", seed=0)
    assert audio_aug.band_energy(white, 0, 1000) < 0.2


def test_telephone_removes_energy_outside_the_band():
    rng = np.random.default_rng(0)
    wide = (0.1 * rng.standard_normal(SR * 2)).astype(np.float32)  # flat 0-8 kHz
    assert audio_aug.band_energy(wide, 4000, 8000) > 0.4
    phone = audio_aug.telephone(wide)
    assert len(phone) == len(wide) and phone.dtype == np.float32
    assert audio_aug.band_energy(phone, 300, 3400) > 0.95
    assert audio_aug.band_energy(phone, 0, 200) < 0.01
    assert audio_aug.band_energy(phone, 4000, 8000) < 0.001


def test_telephone_keeps_in_band_speech():
    t = np.arange(SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    phone = audio_aug.telephone(tone)
    core = slice(SR // 10, -SR // 10)
    ratio = audio_aug.power(phone[core]) / audio_aug.power(tone[core])
    assert 0.7 < ratio < 1.3


def test_save_and_load_round_trip(tmp_path):
    clean = speechy(1.0)
    path = audio_aug.save(tmp_path / "x.wav", clean)
    back = audio_aug.load(path)
    assert len(back) == len(clean)
    assert np.max(np.abs(back - clean)) < 1e-3
    assert len(audio_aug.load(path, sr=8000)) == len(clean) // 2
