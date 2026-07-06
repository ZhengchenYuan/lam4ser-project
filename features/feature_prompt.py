"""
Convert numeric acoustic features into short textual descriptions.

Main purpose:
    numeric acoustic features
    -> textual feature tokens
    -> GPT-2 tokenizer
    -> input_ids
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict

import numpy as np

BASELINE_FEATURE_KEYS = [
    "pitch_mean",
    "pitch_std",
    "energy_mean",
    "energy_std",
    "duration",
    "tempo",
]


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


def compute_feature_baseline(feature_dicts):
    """
    Compute simple mean/std acoustic baselines from enrollment features.
    """
    baseline = {}

    for key in BASELINE_FEATURE_KEYS:
        values = [
            float(features.get(key, 0.0))
            for features in feature_dicts
            if features.get(key, None) is not None
            and np.isfinite(features.get(key, 0.0))
        ]

        if not values:
            baseline[key] = {"mean": 0.0, "std": 1.0}
            continue

        values = np.array(values, dtype=float)
        std = float(values.std())
        if std < 1e-6:
            std = 1.0

        baseline[key] = {
            "mean": float(values.mean()),
            "std": std,
        }

    return baseline


def acoustic_features_to_global_caption(features: Dict[str, float]) -> str:
    """
    Convert absolute acoustic features into a natural-language target caption.
    """
    pitch = _describe_pitch(features.get("pitch_mean", 0.0))
    energy = _describe_energy(features.get("energy_mean", 0.0))
    duration = _describe_duration(features.get("duration", 0.0))
    rhythm = _describe_rhythm(features.get("tempo", 0.0))

    cues = _join_cues([
        _quality_phrase(pitch),
        _quality_phrase(energy),
        _quality_phrase(rhythm),
        _quality_phrase(duration),
    ])

    return f"The utterance has {cues}."


def acoustic_features_to_speaker_relative_caption(
    features: Dict[str, float],
    baseline,
    threshold: float = 0.5,
    baseline_label: str = "neutral baseline",
) -> str:
    """
    Convert acoustic features into a speaker-relative target caption.
    """
    cues = _join_cues([
        _relative_feature_phrase(
            features,
            baseline,
            "pitch_mean",
            lower="relatively lower pitch",
            similar=f"pitch close to this speaker's {baseline_label}",
            higher="relatively higher pitch",
            threshold=threshold,
        ),
        _relative_feature_phrase(
            features,
            baseline,
            "energy_mean",
            lower="relatively lower energy",
            similar=f"energy close to this speaker's {baseline_label}",
            higher="relatively higher energy",
            threshold=threshold,
        ),
        _relative_feature_phrase(
            features,
            baseline,
            "tempo",
            lower="a slower rhythm",
            similar=f"a rhythm close to this speaker's {baseline_label}",
            higher="a faster rhythm",
            threshold=threshold,
        ),
        _relative_feature_phrase(
            features,
            baseline,
            "duration",
            lower="a shorter duration",
            similar=f"a duration close to this speaker's {baseline_label}",
            higher="a longer duration",
            threshold=threshold,
        ),
    ])

    return (
        f"Compared with this speaker's {baseline_label}, "
        f"the utterance has {cues}."
    )


def acoustic_features_to_speaker_relative_cues(
    features: Dict[str, float],
    baseline,
    threshold: float = 0.5,
    baseline_label: str = "neutral baseline",
) -> str:
    """
    Convert acoustic features into compact speaker-relative prompt cues.
    """
    return _join_cues([
        _relative_feature_phrase(
            features,
            baseline,
            "pitch_mean",
            lower=f"lower pitch than this speaker's {baseline_label}",
            similar=f"pitch close to this speaker's {baseline_label}",
            higher=f"higher pitch than this speaker's {baseline_label}",
            threshold=threshold,
        ),
        _relative_feature_phrase(
            features,
            baseline,
            "energy_mean",
            lower=f"lower energy than this speaker's {baseline_label}",
            similar=f"energy close to this speaker's {baseline_label}",
            higher=f"higher energy than this speaker's {baseline_label}",
            threshold=threshold,
        ),
        _relative_feature_phrase(
            features,
            baseline,
            "tempo",
            lower=f"slower rhythm than this speaker's {baseline_label}",
            similar=f"rhythm close to this speaker's {baseline_label}",
            higher=f"faster rhythm than this speaker's {baseline_label}",
            threshold=threshold,
        ),
        _relative_feature_phrase(
            features,
            baseline,
            "duration",
            lower=f"shorter duration than this speaker's {baseline_label}",
            similar=f"duration close to this speaker's {baseline_label}",
            higher=f"longer duration than this speaker's {baseline_label}",
            threshold=threshold,
        ),
    ])


def speaker_relative_evidence_sentence(
    features: Dict[str, float],
    baseline,
    label: str,
    threshold: float = 0.5,
    baseline_label: str = "neutral baseline",
) -> str:
    """
    Build a short evidence sentence from the most salient speaker-relative cues.
    """
    cues = _speaker_relative_evidence_cues(
        features,
        baseline,
        threshold,
        baseline_label,
    )
    cue_text = _join_cues([cue["phrase"] for cue in cues])
    verb = "suggests" if len(cues) == 1 and cues[0].get("singular") else "suggest"
    interpretation = _interpret_evidence_cues(cues)

    return f"{cue_text} {verb} {interpretation}, supporting {label}."


def _speaker_relative_evidence_cues(features, baseline, threshold, baseline_label):
    cue_specs = [
        (
            "pitch_mean",
            f"lower pitch than this speaker's {baseline_label}",
            f"higher pitch than this speaker's {baseline_label}",
            "pitch",
        ),
        (
            "energy_mean",
            f"lower energy than this speaker's {baseline_label}",
            f"higher energy than this speaker's {baseline_label}",
            "energy",
        ),
        (
            "tempo",
            f"slower rhythm than this speaker's {baseline_label}",
            f"faster rhythm than this speaker's {baseline_label}",
            "rhythm",
        ),
        (
            "duration",
            f"shorter duration than this speaker's {baseline_label}",
            f"longer duration than this speaker's {baseline_label}",
            "duration",
        ),
    ]

    cues = []
    for key, lower, higher, name in cue_specs:
        stats = baseline.get(key, {"mean": 0.0, "std": 1.0})
        z = _zscore(
            features.get(key, 0.0),
            stats.get("mean", 0.0),
            stats.get("std", 1.0),
        )

        if abs(z) < threshold:
            continue

        cues.append({
            "name": name,
            "z": z,
            "phrase": higher if z > 0 else lower,
        })

    cues.sort(key=lambda cue: abs(cue["z"]), reverse=True)

    if cues:
        return cues[:2]

    return [{
        "name": "delivery",
        "z": 0.0,
        "phrase": f"prosody close to this speaker's {baseline_label}",
        "singular": True,
    }]


def _interpret_evidence_cues(cues):
    high_activation = any(
        cue["name"] in {"energy", "rhythm", "pitch"} and cue["z"] > 0
        for cue in cues
    )
    low_activation = any(
        cue["name"] in {"energy", "rhythm", "pitch"} and cue["z"] < 0
        for cue in cues
    )
    longer = any(cue["name"] == "duration" and cue["z"] > 0 for cue in cues)
    baseline_like = all(abs(cue["z"]) < 1e-6 for cue in cues)

    if baseline_like:
        return "a steady speaker-relative delivery"
    if high_activation:
        return "stronger activation and expressiveness"
    if low_activation and longer:
        return "subdued and sustained delivery"
    if low_activation:
        return "reduced activation and subdued delivery"

    return "a distinct speaker-relative prosodic pattern"


def emotion_reasoning_sentence(
    label: str,
    baseline_label: str | None = None,
) -> str:
    """
    Template-based target-side emotion reasoning.
    """
    neutral_reasoning = (
        f"These cues are close to this speaker's {baseline_label}, which supports "
        "neutral."
        if baseline_label == "neutral baseline"
        else "These cues do not strongly deviate from this speaker-specific "
        "enrollment average, which supports neutral."
    )

    reasoning = {
        "anger": (
            "These cues suggest stronger emotional activation and tense delivery, "
            "which supports anger."
        ),
        "happiness": (
            "These cues suggest lively or energetic delivery, which supports "
            "happiness."
        ),
        "sadness": (
            "These cues suggest subdued delivery and reduced activation, which "
            "supports sadness."
        ),
        "boredom": (
            "These cues suggest low activation and flat or slow delivery, which "
            "supports boredom."
        ),
        "neutral": neutral_reasoning,
        "fear": (
            "These cues suggest activated or tense delivery with elevated arousal, "
            "which supports fear."
        ),
        "disgust": (
            "These cues suggest tense, harsh, or restrained delivery, which "
            "supports disgust."
        ),
    }

    return reasoning.get(
        label,
        f"These cues support the {label} emotion label.",
    )


def _relative_feature_phrase(
    features,
    baseline,
    key,
    lower,
    similar,
    higher,
    threshold,
):
    stats = baseline.get(key, {"mean": 0.0, "std": 1.0})
    z = _zscore(
        features.get(key, 0.0),
        stats.get("mean", 0.0),
        stats.get("std", 1.0),
    )

    if z > threshold:
        return higher
    if z < -threshold:
        return lower
    return similar


def _describe_rhythm(tempo: float) -> str:
    rhythm = _describe_tempo(tempo)
    if rhythm == "slow tempo":
        return "slow rhythm"
    if rhythm == "medium tempo":
        return "medium rhythm"
    if rhythm == "fast tempo":
        return "fast rhythm"
    return "unknown rhythm"


def _quality_phrase(text: str) -> str:
    if text.startswith("unknown"):
        return None
    if text in {"short duration", "medium duration", "long duration"}:
        return f"a {text}"
    return text


def _join_cues(cues) -> str:
    cues = [cue for cue in cues if cue]

    if not cues:
        return "limited measurable acoustic cues"
    if len(cues) == 1:
        return cues[0]
    if len(cues) == 2:
        return f"{cues[0]} and {cues[1]}"

    return f"{', '.join(cues[:-1])}, and {cues[-1]}"




def get_speaker_id_from_path(path: str) -> str:
    """
    EmoDB filenames start with the speaker ID, e.g. 03a01Fa.wav.
    AIBO filenames start with school and speaker ID, e.g. Mont_01_000_00.wav.
    """
    basename = os.path.splitext(os.path.basename(str(path)))[0]
    if len(basename) < 2:
        return "unknown"
    if basename[0].isdigit():
        return basename[:2]

    parts = basename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return basename[:2]


def compute_speaker_feature_stats(acoustic_feature_cache, file_paths):
    """
    Compute speaker-wise mean/std for selected acoustic features.

    acoustic_feature_cache:
        list of feature dictionaries, indexed by sample index.

    file_paths:
        list of wav paths, same order as acoustic_feature_cache.
    """
    speaker_values = defaultdict(lambda: defaultdict(list))

    keys = [
        "pitch_mean",
        "pitch_std",
        "energy_mean",
        "energy_std",
    ]

    for idx, path in enumerate(file_paths):
        speaker = get_speaker_id_from_path(path)
        features = acoustic_feature_cache[idx]

        for key in keys:
            value = features.get(key, 0.0)

            if value is not None and np.isfinite(value):
                speaker_values[speaker][key].append(float(value))

    speaker_stats = {}

    for speaker, values_by_key in speaker_values.items():
        speaker_stats[speaker] = {}

        for key in keys:
            values = np.array(values_by_key.get(key, []), dtype=float)

            if len(values) == 0:
                speaker_stats[speaker][key] = {"mean": 0.0, "std": 1.0}
                continue

            mean = float(values.mean())
            std = float(values.std())

            if std < 1e-6:
                std = 1.0

            speaker_stats[speaker][key] = {
                "mean": mean,
                "std": std,
            }

    return speaker_stats


def _zscore(value, mean, std):
    if value is None or not np.isfinite(value):
        return 0.0
    if std < 1e-6:
        return 0.0
    return (float(value) - float(mean)) / float(std)


def _relative_descriptor(z, name):
    if z <= -0.75:
        return f"{name} lower than speaker baseline"
    elif z >= 0.75:
        return f"{name} higher than speaker baseline"
    else:
        return f"{name} around speaker baseline"


def acoustic_features_to_speaker_relative_text(features, speaker_id, speaker_stats):
    """
    Verbalize pitch/energy relative to each speaker's own baseline.

    Duration and tempo are still verbalized with the original absolute rules,
    because they are less directly speaker-baseline dependent.
    """
    stats = speaker_stats.get(speaker_id, {})

    pitch_mean_z = _zscore(
        features.get("pitch_mean", 0.0),
        stats.get("pitch_mean", {}).get("mean", 0.0),
        stats.get("pitch_mean", {}).get("std", 1.0),
    )

    pitch_std_z = _zscore(
        features.get("pitch_std", 0.0),
        stats.get("pitch_std", {}).get("mean", 0.0),
        stats.get("pitch_std", {}).get("std", 1.0),
    )

    energy_mean_z = _zscore(
        features.get("energy_mean", 0.0),
        stats.get("energy_mean", {}).get("mean", 0.0),
        stats.get("energy_mean", {}).get("std", 1.0),
    )

    energy_std_z = _zscore(
        features.get("energy_std", 0.0),
        stats.get("energy_std", {}).get("mean", 0.0),
        stats.get("energy_std", {}).get("std", 1.0),
    )

    parts = [
        _relative_descriptor(pitch_mean_z, "pitch"),
        _relative_descriptor(pitch_std_z, "pitch variation"),
        _relative_descriptor(energy_mean_z, "energy"),
        _relative_descriptor(energy_std_z, "energy variation"),
    ]

    # Keep the old absolute verbalization for duration / tempo by reusing existing function.
    # This assumes your existing acoustic_features_to_text(...) returns a comma-separated string.
    absolute_text = acoustic_features_to_text(features)
    for phrase in absolute_text.split(","):
        phrase = phrase.strip()
        if "duration" in phrase or "tempo" in phrase:
            parts.append(phrase)

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
