"""
Acoustic feature extraction for LAM4SER.

This module converts a raw wav file into a small numeric feature dictionary.
The output can later be verbalized into text and inserted into the GPT-2 prompt.

Main purpose:
    wav path
    -> pitch / energy / duration / tempo
    -> numeric feature dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def extract_acoustic_features(
    wav_path: str | Path,
    sr: int = 16000,
) -> Dict[str, float]:
    """
    Extract simple acoustic features from a wav file.

    Extracted features:
        - duration
        - pitch_mean
        - pitch_std
        - energy_mean
        - energy_std
        - tempo

    Args:
        wav_path:
            Path to the original wav file.

        sr:
            Target sampling rate for librosa loading.

    Returns:
        Dictionary with acoustic features.
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for acoustic feature extraction. "
            "Install it with: pip install librosa"
        ) from exc

    wav_path = Path(wav_path)

    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    y, sr = librosa.load(str(wav_path), sr=sr, mono=True)

    if y.size == 0:
        return _empty_features()

    duration = float(librosa.get_duration(y=y, sr=sr))

    # ------------------------------------------------------------------
    # Pitch extraction
    # ------------------------------------------------------------------
    # librosa.pyin is more stable than naive pitch estimation, but it can
    # return many NaNs. We ignore unvoiced frames.
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced_f0 = f0[~np.isnan(f0)]

        if voiced_f0.size > 0:
            pitch_mean = float(np.mean(voiced_f0))
            pitch_std = float(np.std(voiced_f0))
        else:
            pitch_mean = 0.0
            pitch_std = 0.0

    except Exception:
        # Keep the pipeline robust. If pitch extraction fails, do not crash
        # the whole training run.
        pitch_mean = 0.0
        pitch_std = 0.0

    # ------------------------------------------------------------------
    # Energy extraction
    # ------------------------------------------------------------------
    rms = librosa.feature.rms(y=y)[0]

    if rms.size > 0:
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
    else:
        energy_mean = 0.0
        energy_std = 0.0

    # ------------------------------------------------------------------
    # Tempo / onset-rate proxy
    # ------------------------------------------------------------------
    try:
        tempo_arr = librosa.beat.tempo(y=y, sr=sr)
        tempo = float(tempo_arr[0]) if len(tempo_arr) > 0 else 0.0
    except Exception:
        tempo = 0.0

    return {
        "duration": duration,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "tempo": tempo,
    }


def _empty_features() -> Dict[str, float]:
    """
    Return a safe all-zero feature dictionary.
    """
    return {
        "duration": 0.0,
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "energy_mean": 0.0,
        "energy_std": 0.0,
        "tempo": 0.0,
    }
