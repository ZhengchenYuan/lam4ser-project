"""
Convert numeric acoustic features into short textual descriptions.

Main purpose:
    numeric acoustic features
    -> textual feature tokens
    -> GPT-2 tokenizer
    -> input_ids
"""

from __future__ import annotations

from typing import Dict


def acoustic_features_to_text(features: Dict[str, float]) -> str:
    """
    Convert numeric acoustic features into a short text description.

    Example:
        {
            "pitch_mean": 260,
            "energy_mean": 0.09,
            "duration": 1.8,
            "tempo": 120,
        }

        -> "high pitch, high energy, short duration, medium tempo"

    Args:
        features:
            Acoustic feature dictionary from extract_acoustic_features.

    Returns:
        Short textual acoustic description.
    """
    pitch = _describe_pitch(features.get("pitch_mean", 0.0))
    pitch_var = _describe_pitch_variation(features.get("pitch_std", 0.0))
    energy = _describe_energy(features.get("energy_mean", 0.0))
    energy_var = _describe_energy_variation(features.get("energy_std", 0.0))
    duration = _describe_duration(features.get("duration", 0.0))
    tempo = _describe_tempo(features.get("tempo", 0.0))

    parts = [
        pitch,
        pitch_var,
        energy,
        energy_var,
        duration,
        tempo,
    ]

    return ", ".join(parts)


def _describe_pitch(pitch_mean: float) -> str:
    """
    Describe mean pitch.

    Thresholds are intentionally simple for the first implementation.
    Later you can replace them with dataset-level quantiles.
    """
    if pitch_mean <= 0:
        return "unknown pitch"
    if pitch_mean < 160:
        return "low pitch"
    if pitch_mean < 240:
        return "medium pitch"
    return "high pitch"


def _describe_pitch_variation(pitch_std: float) -> str:
    """
    Describe pitch variation.
    """
    if pitch_std <= 0:
        return "unknown pitch variation"
    if pitch_std < 25:
        return "stable pitch"
    if pitch_std < 60:
        return "moderate pitch variation"
    return "large pitch variation"


def _describe_energy(energy_mean: float) -> str:
    """
    Describe mean energy.
    """
    if energy_mean <= 0:
        return "unknown energy"
    if energy_mean < 0.03:
        return "low energy"
    if energy_mean < 0.08:
        return "medium energy"
    return "high energy"


def _describe_energy_variation(energy_std: float) -> str:
    """
    Describe energy variation.
    """
    if energy_std <= 0:
        return "unknown energy variation"
    if energy_std < 0.01:
        return "stable energy"
    if energy_std < 0.03:
        return "moderate energy variation"
    return "large energy variation"


def _describe_duration(duration: float) -> str:
    """
    Describe utterance duration.
    """
    if duration <= 0:
        return "unknown duration"
    if duration < 2.0:
        return "short duration"
    if duration < 5.0:
        return "medium duration"
    return "long duration"


def _describe_tempo(tempo: float) -> str:
    """
    Describe tempo.

    For short speech utterances, tempo estimation may be noisy.
    This is acceptable for the first feature-token baseline.
    """
    if tempo <= 0:
        return "unknown tempo"
    if tempo < 90:
        return "slow tempo"
    if tempo < 140:
        return "medium tempo"
    return "fast tempo"
