"""Generate deterministic, offline label-blind acoustic rationales for AIBO.

The generator deliberately uses only evidence available in this repository.
No external model or API is used.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import speaker_independent_split
from data.dataset_configs import get_dataset_config
from data.generation_dataset import EmoDBGenerationDataset
from features.feature_prompt import acoustic_features_to_speaker_relative_caption


PROMPT_VERSION = "aibo_label_blind_rationale_v1"


def _zscore(value: float, stats: dict) -> float:
    std = max(float(stats.get("std", 1.0)), 1e-6)
    return (float(value) - float(stats.get("mean", 0.0))) / std


def _level(z: float, lower: str, similar: str, higher: str) -> str:
    if z < -0.5:
        return lower
    if z > 0.5:
        return higher
    return similar


def _deterministic_think(features: dict, baseline: dict) -> str:
    cue_specs = (
        ("pitch", "pitch_mean", "lower", "baseline-like", "higher"),
        ("energy", "energy_mean", "lower", "baseline-like", "higher"),
        ("rhythm", "tempo", "slower", "baseline-like", "faster"),
        ("duration", "duration", "shorter", "baseline-like", "longer"),
    )
    cues = []
    for name, key, lower, similar, higher in cue_specs:
        z = _zscore(features.get(key, 0.0), baseline.get(key, {}))
        cues.append((name, z, _level(z, lower, similar, higher)))

    observation = ", ".join(f"{level} {name}" for name, _, level in cues)
    directions = [
        1 if z > 0.5 else -1 if z < -0.5 else 0
        for name, z, _ in cues
        if name in {"pitch", "energy", "rhythm"}
    ]
    active_directions = {direction for direction in directions if direction != 0}
    if not active_directions:
        pattern = (
            "the measured pitch, energy, and rhythm remain broadly "
            "baseline-like, indicating no clear activation shift."
        )
    elif active_directions == {1}:
        pattern = (
            "the available pitch, energy, and rhythm cues consistently point "
            "toward increased vocal activation."
        )
    elif active_directions == {-1}:
        pattern = (
            "the available pitch, energy, and rhythm cues consistently point "
            "toward reduced vocal activation."
        )
    else:
        pattern = (
            "the available pitch, energy, and rhythm cues point in different "
            "directions, so the activation evidence is mixed."
        )

    return (
        f"First, relative to this speaker's neutral baseline, the utterance shows "
        f"{observation}. Second, {pattern} These acoustic measurements "
        f"characterize how the utterance was delivered, but without lexical or "
        f"contextual information they are not uniquely diagnostic of one emotion."
    )


def _rationale_baseline(baseline: dict, parameters: dict) -> dict:
    normalized = {key: dict(stats) for key, stats in baseline.items()}
    for key, parameter_stats in parameters.items():
        residual_std = math.sqrt(max(parameter_stats["residual_variance"], 0.0))
        feature_stats = normalized.setdefault(key, {})
        if residual_std >= 1e-6:
            feature_stats["std"] = residual_std
    return normalized


def _existing_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fp:
        records = [json.loads(line) for line in fp if line.strip()]
    for record in records:
        if record.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(
                f"Refusing to mix annotation versions in {path}; use a new output."
            )
    return {str(record["sample_id"]) for record in records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="wavlm-large")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = get_dataset_config("aibo")
    embeddings_path = f"embeddings/aibo_{args.encoder}_embeddings.pt"
    dataset = EmoDBGenerationDataset(
        embeddings_path=embeddings_path,
        prompt_type="speaker_reasoning_generation",
        speaker_baseline_mode="neutral",
    )
    train_idx, val_idx, test_idx = speaker_independent_split(
        dataset,
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
    )
    baseline_summary = dataset.apply_baseline_estimation("mixed_effects", train_idx)
    split_by_idx = {
        **{idx: "train" for idx in train_idx},
        **{idx: "validation" for idx in val_idx},
        **{idx: "test" for idx in test_idx},
    }

    completed = _existing_ids(args.output)
    written = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as fp:
        for sample_idx, real_idx in enumerate(dataset.sample_indices):
            sample_id = os.path.splitext(
                os.path.basename(str(dataset.all_file_paths[real_idx]))
            )[0]
            if sample_id in completed:
                continue
            features = dataset.acoustic_feature_cache[real_idx]
            speaker_id = dataset._speaker_for_real_idx(real_idx)
            baseline = _rationale_baseline(
                dataset.speaker_baselines[speaker_id],
                baseline_summary["parameters"],
            )
            relative_caption = acoustic_features_to_speaker_relative_caption(
                features,
                baseline,
                baseline_label=dataset._baseline_label_for_speaker(speaker_id),
            )
            answer = dataset._label_to_text(dataset.labels[real_idx])
            evidence = {
                "speaker_id": speaker_id,
                "speaker_relative_prosody": relative_caption,
                "numeric_prosody": {
                    key: features.get(key)
                    for key in (
                        "pitch_mean", "pitch_std", "energy_mean", "energy_std",
                        "tempo", "duration",
                    )
                },
                "unavailable": [
                    "transcript", "age", "gender", "word_stress",
                    "intonation_contour",
                ],
            }
            think = _deterministic_think(features, baseline)

            record = {
                "sample_id": sample_id,
                "split": split_by_idx[sample_idx],
                "answer": answer,
                "think": think,
                "annotation_model": "deterministic-local",
                "prompt_version": PROMPT_VERSION,
                "evidence": evidence,
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()
            written += 1
            if args.limit is not None and written >= args.limit:
                break

    print(f"Wrote {written} annotations to {args.output}")


if __name__ == "__main__":
    main()
