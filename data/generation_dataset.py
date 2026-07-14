import os
from collections import defaultdict
import torch
from torch.utils.data import Dataset

from data.prompts import get_prompt
from data.tokenizer_utils import build_generation_tokenizer
from features.acoustic_features import extract_acoustic_features
from features.feature_prompt import (
    acoustic_features_to_global_caption,
    acoustic_features_to_speaker_relative_caption,
    acoustic_features_to_speaker_relative_cues,
    acoustic_features_to_text,
    compute_feature_baseline,
    emotion_reasoning_sentence,
    speaker_relative_evidence_sentence,
)


MIXED_EFFECTS_BASELINE_FEATURE_KEYS = (
    "pitch_mean",
    "energy_mean",
    "tempo",
    "duration",
)

BASELINE_ESTIMATION_MODES = (
    "speaker_neutral",
    "mixed_effects",
)


def print_baseline_estimation_summary(summary: dict):
    print("Baseline estimation summary:")
    print(f"  Mode: {summary['mode']}")

    if summary["mode"] != "mixed_effects":
        return

    print(f"  Train neutral samples: {summary['train_neutral_samples']}")
    for feature_name, stats in summary["parameters"].items():
        print(
            f"  {feature_name}: "
            f"mu={stats['mu']:.6f}, "
            f"sigma_speaker^2={stats['speaker_variance']:.6f}, "
            f"sigma_residual^2={stats['residual_variance']:.6f}"
        )

    if summary["fallback_speakers"]:
        print(
            "  Fallback to global mean for speakers without neutral enrollment: "
            f"{summary['fallback_speakers']}"
        )
    else:
        print("  Fallback to global mean: none")

GENERATION_PROMPT_TYPES = (
    "generation",
    "feature_generation",
    "answer_generation",
    "speaker_feature_answer_generation",
    "speaker_feature_answer_caption_generation",
    "speaker_feature_answer_evidence_generation",
    "reasoning_generation_global",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
    "speaker_acoustic_cue_generation",
)

ANSWER_TAG_PROMPT_TYPES = (
    "answer_generation",
    "speaker_feature_answer_generation",
    "speaker_feature_answer_caption_generation",
    "speaker_feature_answer_evidence_generation",
)

CAPTION_TARGET_PROMPT_TYPES = (
    "speaker_feature_answer_caption_generation",
)

EVIDENCE_TARGET_PROMPT_TYPES = (
    "speaker_feature_answer_evidence_generation",
)

REASONING_PROMPT_TYPES = (
    "reasoning_generation_global",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
)

SPEAKER_REASONING_PROMPT_TYPES = (
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
)

SPEAKER_BASELINE_PROMPT_TYPES = (
    "speaker_feature_answer_generation",
    "speaker_feature_answer_caption_generation",
    "speaker_feature_answer_evidence_generation",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
    "speaker_acoustic_cue_generation",
)

ACOUSTIC_CUE_PROMPT_TYPES = (
    "speaker_acoustic_cue_generation",
)

SPEAKER_BASELINE_MODES = (
    "neutral",
    "emotion_balanced",
)


def extract_speaker_id(file_path: str) -> str:
    """
    EmoDB filenames start with a two-digit speaker ID, e.g. 03a01Fa.wav.
    AIBO chunk filenames start with school and speaker ID, e.g.
    Mont_01_000_00.wav.
    """
    basename = os.path.splitext(os.path.basename(file_path))[0]
    if len(basename) < 2:
        return "unknown"

    if basename[0].isdigit():
        return basename[:2]

    parts = basename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return basename[:2]


class EmoDBGenerationDataset(Dataset):
    def __init__(
        self,
        embeddings_path: str,
        prompt_type: str = "generation",
        max_length: int = 96,
        answer_loss_weight: float = 5.0,
        evidence_loss_weight: float = 0.3,
        speaker_baseline_mode: str = "neutral",
        disable_input_cue_text: bool = False,
    ):
        if not os.path.exists(embeddings_path):
            print(
                f"ERROR: '{embeddings_path}' not found. "
                "Run models/audio_encoder/preprocessing.py first to generate the embeddings file."
            )
            raise FileNotFoundError(f"'{embeddings_path}' not found")

        if prompt_type not in GENERATION_PROMPT_TYPES:
            raise ValueError(
                "EmoDBGenerationDataset only supports prompt_type in "
                f"{GENERATION_PROMPT_TYPES}. "
                f"Got: {prompt_type}"
            )

        self.embeddings_path = embeddings_path
        self.prompt_type = prompt_type
        self.max_length = max_length
        self.answer_loss_weight = answer_loss_weight
        self.evidence_loss_weight = evidence_loss_weight
        if speaker_baseline_mode not in SPEAKER_BASELINE_MODES:
            raise ValueError(
                f"speaker_baseline_mode must be one of {SPEAKER_BASELINE_MODES}. "
                f"Got: {speaker_baseline_mode}"
            )
        self.speaker_baseline_mode = speaker_baseline_mode
        self.disable_input_cue_text = disable_input_cue_text
        self.use_feature_prompt = "feature" in prompt_type
        self.use_answer_tag_target = prompt_type in ANSWER_TAG_PROMPT_TYPES
        self.use_caption_target = prompt_type in CAPTION_TARGET_PROMPT_TYPES
        self.use_evidence_target = prompt_type in EVIDENCE_TARGET_PROMPT_TYPES
        self.use_reasoning_target = prompt_type in REASONING_PROMPT_TYPES
        self.use_speaker_reasoning = prompt_type in SPEAKER_REASONING_PROMPT_TYPES
        self.use_speaker_baseline = prompt_type in SPEAKER_BASELINE_PROMPT_TYPES
        self.use_acoustic_cue_target = prompt_type in ACOUSTIC_CUE_PROMPT_TYPES

        data = torch.load(embeddings_path, weights_only=False)

        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.label2idx = data["label2idx"]
        self.idx2label = data["idx2label"]
        self.label_names = [self.idx2label[i] for i in range(len(self.idx2label))]

        self.all_file_paths = None
        self.file_paths = None
        self.speaker_ids = None

        for key in ("file_paths", "paths", "files"):
            if key in data:
                self.all_file_paths = data[key]
                break

        if self.all_file_paths is None:
            print(
                "WARNING: No file paths found in embeddings file.\n"
                "Speaker-independent splitting is not available.\n"
                "Falling back to random 70/15/15 split."
            )
        else:
            self.file_paths = list(self.all_file_paths)
            self.speaker_ids = [extract_speaker_id(p) for p in self.file_paths]

        needs_acoustic_features = (
            self.use_feature_prompt
            or self.use_reasoning_target
            or self.use_acoustic_cue_target
        )

        if needs_acoustic_features and self.file_paths is None:
            raise ValueError(
                f"{prompt_type} requires wav file paths, but no key among "
                "('file_paths', 'paths', 'files') was found in the embeddings file."
            )

        self.tokenizer = build_generation_tokenizer(verbose=True)

        self.acoustic_feature_cache = None
        self.speaker_baselines = {}
        self.enrollment_indices = set()
        self.neutral_fallback_speakers = []
        self.sample_indices = list(range(len(self.embeddings)))

        if needs_acoustic_features:
            cache_path = embeddings_path.replace(
                "_embeddings.pt",
                "_acoustic_features.pt",
            )

            if os.path.exists(cache_path):
                print(f"Loading cached acoustic features from: {cache_path}")
                self.acoustic_feature_cache = torch.load(
                    cache_path,
                    weights_only=False,
                )
            else:
                print("Extracting acoustic features from wav files...")
                self.acoustic_feature_cache = []

                for i, wav_path in enumerate(self.file_paths):
                    if i % 50 == 0:
                        print(
                            f"  Extracting acoustic features: "
                            f"{i}/{len(self.file_paths)}"
                        )

                    feature_dict = extract_acoustic_features(wav_path)
                    self.acoustic_feature_cache.append(feature_dict)

                torch.save(self.acoustic_feature_cache, cache_path)
                print(f"Saved acoustic feature cache to: {cache_path}")

        if self.use_speaker_baseline:
            self._build_enrollment_sets()

        self._rebuild_generation_samples()

    def _rebuild_generation_samples(self):
        self.input_ids_list = []
        self.lm_labels_list = []
        self.loss_weights_list = []
        self.answer_loss_masks_list = []
        self.class_labels_list = []

        for idx in range(len(self.sample_indices)):
            input_ids, lm_labels, loss_weights, answer_loss_mask = (
                self._build_generation_sample(idx)
            )
            self.input_ids_list.append(input_ids)
            self.lm_labels_list.append(lm_labels)
            self.loss_weights_list.append(loss_weights)
            self.answer_loss_masks_list.append(answer_loss_mask)
            self.class_labels_list.append(
                torch.tensor(self.labels[self.sample_indices[idx]], dtype=torch.long)
            )

    def _build_enrollment_sets(self):
        by_speaker_label = defaultdict(list)

        for idx, path in enumerate(self.file_paths):
            speaker_id = extract_speaker_id(path)
            label_text = self._label_to_text(self.labels[idx])
            by_speaker_label[(speaker_id, label_text)].append(idx)

        for indices in by_speaker_label.values():
            indices.sort(key=lambda i: os.path.basename(str(self.file_paths[i])))

        if self.speaker_baseline_mode == "neutral":
            self._select_neutral_enrollment(by_speaker_label)
        elif self.speaker_baseline_mode == "emotion_balanced":
            self._select_emotion_balanced_enrollment(by_speaker_label)
        else:
            raise ValueError(
                f"Unknown speaker baseline mode: {self.speaker_baseline_mode}"
            )

        self.sample_indices = [
            idx
            for idx in range(len(self.embeddings))
            if idx not in self.enrollment_indices
        ]

        enrollment_by_speaker = defaultdict(list)
        for idx in sorted(self.enrollment_indices):
            speaker_id = extract_speaker_id(self.file_paths[idx])
            enrollment_by_speaker[speaker_id].append(self.acoustic_feature_cache[idx])

        self.speaker_baselines = {
            speaker_id: compute_feature_baseline(features)
            for speaker_id, features in enrollment_by_speaker.items()
        }

        self._print_enrollment_summary(
            by_speaker_label=by_speaker_label,
            enrollment_by_speaker=enrollment_by_speaker,
        )

    def _select_emotion_balanced_enrollment(self, by_speaker_label):
        for indices in by_speaker_label.values():
            self.enrollment_indices.add(indices[0])

    def _select_neutral_enrollment(self, by_speaker_label):
        labels_by_speaker = defaultdict(dict)
        for (speaker_id, label_text), indices in by_speaker_label.items():
            labels_by_speaker[speaker_id][label_text] = indices

        fallback_speakers = []
        for speaker_id, labels in labels_by_speaker.items():
            neutral_indices = labels.get("neutral")
            if neutral_indices:
                self.enrollment_indices.add(neutral_indices[0])
                continue

            fallback_speakers.append(speaker_id)
            for indices in labels.values():
                self.enrollment_indices.add(indices[0])

        self.neutral_fallback_speakers = sorted(fallback_speakers)

    def _print_enrollment_summary(self, by_speaker_label, enrollment_by_speaker):
        target_speakers = [
            extract_speaker_id(self.file_paths[idx])
            for idx in self.sample_indices
        ]
        expected_labels = set(self._label_to_text(label) for label in self.labels)
        labels_by_speaker = defaultdict(set)
        for speaker_id, label_text in by_speaker_label:
            labels_by_speaker[speaker_id].add(label_text)

        partial_enrollment_speakers = sorted(
            speaker_id
            for speaker_id, labels in labels_by_speaker.items()
            if labels != expected_labels
        )
        speakers_without_targets = sorted(
            set(enrollment_by_speaker) - set(target_speakers)
        )

        self.file_paths = [self.all_file_paths[idx] for idx in self.sample_indices]
        self.speaker_ids = [extract_speaker_id(p) for p in self.file_paths]

        print("Speaker enrollment summary:")
        print(f"  Baseline mode:      {self.speaker_baseline_mode}")
        if self.speaker_baseline_mode == "neutral":
            print("  Baseline type:      neutral-only where available")
        else:
            print("  Baseline type:      one utterance per emotion class")
        print(f"  Speakers:           {len(enrollment_by_speaker)}")
        print(f"  Enrollment samples: {len(self.enrollment_indices)}")
        print(f"  Target samples:     {len(self.sample_indices)}")
        fallback_messages = 0

        if self.speaker_baseline_mode == "neutral":
            if self.neutral_fallback_speakers:
                print(
                    "  Neutral fallback: speakers without neutral samples; "
                    "using emotion-balanced enrollment: "
                    f"{self.neutral_fallback_speakers}"
                )
                fallback_messages += 1
            else:
                print("  Neutral fallback: none")

        if partial_enrollment_speakers:
            print(
                "  Fallback: speakers without every emotion class in enrollment: "
                f"{partial_enrollment_speakers}"
            )
            fallback_messages += 1

        if speakers_without_targets:
            print(
                "  Fallback: speakers with enrollment but no remaining targets: "
                f"{speakers_without_targets}"
            )
            fallback_messages += 1

        if fallback_messages == 0:
            print("  Fallback: none")

    def _label_to_text(self, label_idx: int) -> str:
        if isinstance(self.idx2label, dict):
            return str(self.idx2label[int(label_idx)])

        return str(self.idx2label[int(label_idx)])

    def _sample_label_text(self, sample_idx: int) -> str:
        real_idx = self.sample_indices[sample_idx]
        return self._label_to_text(self.labels[real_idx])

    def _speaker_for_real_idx(self, real_idx: int) -> str:
        return extract_speaker_id(self.all_file_paths[real_idx])

    def _mean(self, values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    def _variance(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean_value = self._mean(values)
        return sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)

    def _train_neutral_features_by_speaker(self, train_idx):
        features_by_speaker = defaultdict(list)
        for sample_idx in train_idx:
            if self._sample_label_text(sample_idx) != "neutral":
                continue
            real_idx = self.sample_indices[sample_idx]
            speaker_id = self._speaker_for_real_idx(real_idx)
            features_by_speaker[speaker_id].append(self.acoustic_feature_cache[real_idx])
        return features_by_speaker

    def _neutral_enrollment_features_by_speaker(self):
        features_by_speaker = defaultdict(list)
        for real_idx in sorted(self.enrollment_indices):
            if self._label_to_text(self.labels[real_idx]) != "neutral":
                continue
            speaker_id = self._speaker_for_real_idx(real_idx)
            features_by_speaker[speaker_id].append(self.acoustic_feature_cache[real_idx])
        return features_by_speaker

    def _estimate_mixed_effects_parameters(self, train_idx):
        neutral_by_speaker = self._train_neutral_features_by_speaker(train_idx)
        train_neutral_count = sum(
            len(speaker_features)
            for speaker_features in neutral_by_speaker.values()
        )
        if train_neutral_count == 0:
            raise ValueError(
                "mixed_effects baseline estimation requires at least one "
                "train-split neutral sample."
            )

        parameters = {}
        for key in MIXED_EFFECTS_BASELINE_FEATURE_KEYS:
            values_by_speaker = {
                speaker_id: [
                    float(features.get(key, 0.0) or 0.0)
                    for features in speaker_features
                ]
                for speaker_id, speaker_features in neutral_by_speaker.items()
            }
            all_values = [
                value
                for speaker_values in values_by_speaker.values()
                for value in speaker_values
            ]
            global_mean = self._mean(all_values)
            speaker_means = {
                speaker_id: self._mean(speaker_values)
                for speaker_id, speaker_values in values_by_speaker.items()
            }
            residual_sse = sum(
                (value - speaker_means[speaker_id]) ** 2
                for speaker_id, speaker_values in values_by_speaker.items()
                for value in speaker_values
            )
            residual_df = train_neutral_count - len(values_by_speaker)
            residual_variance = residual_sse / residual_df if residual_df > 0 else 0.0
            mean_variance = self._variance(list(speaker_means.values()))
            mean_inverse_n = self._mean([
                1.0 / len(speaker_values)
                for speaker_values in values_by_speaker.values()
            ])
            speaker_variance = max(
                0.0,
                mean_variance - residual_variance * mean_inverse_n,
            )
            parameters[key] = {
                "mu": global_mean,
                "speaker_variance": speaker_variance,
                "residual_variance": residual_variance,
            }

        return parameters, train_neutral_count

    def apply_baseline_estimation(self, mode: str, train_idx) -> dict:
        if mode == "speaker_neutral":
            return {
                "mode": "speaker_neutral",
                "train_neutral_samples": None,
                "parameters": {},
                "fallback_speakers": [],
                "fallback_speaker_count": 0,
            }
        if mode != "mixed_effects":
            raise ValueError(f"Unknown baseline_estimation_mode: {mode}")
        if not self.use_speaker_baseline:
            raise ValueError(
                "mixed_effects baseline estimation requires a speaker-baseline "
                "generation prompt type."
            )

        parameters, train_neutral_count = self._estimate_mixed_effects_parameters(
            train_idx
        )
        neutral_enrollment = self._neutral_enrollment_features_by_speaker()
        target_speakers = sorted(set(self.speaker_ids or []))
        fallback_speakers = []

        for speaker_id in target_speakers:
            speaker_features = neutral_enrollment.get(speaker_id, [])
            if not speaker_features:
                fallback_speakers.append(speaker_id)

            baseline = {
                key: dict(value)
                for key, value in self.speaker_baselines.get(speaker_id, {}).items()
            }
            for key, stats in parameters.items():
                feature_stats = baseline.setdefault(
                    key,
                    {"mean": stats["mu"], "std": 1.0},
                )
                if not speaker_features:
                    feature_stats["mean"] = stats["mu"]
                    continue

                speaker_values = [
                    float(features.get(key, 0.0) or 0.0)
                    for features in speaker_features
                ]
                speaker_mean = self._mean(speaker_values)
                n_s = len(speaker_values)
                denominator = (
                    stats["speaker_variance"]
                    + stats["residual_variance"] / max(n_s, 1)
                )
                lambda_sf = (
                    stats["speaker_variance"] / denominator
                    if denominator > 0.0
                    else 0.0
                )
                feature_stats["mean"] = (
                    stats["mu"] + lambda_sf * (speaker_mean - stats["mu"])
                )

            self.speaker_baselines[speaker_id] = baseline

        self._rebuild_generation_samples()
        return {
            "mode": "mixed_effects",
            "train_neutral_samples": train_neutral_count,
            "parameters": parameters,
            "fallback_speakers": fallback_speakers,
            "fallback_speaker_count": len(fallback_speakers),
        }

    def _baseline_label_for_speaker(self, speaker_id: str) -> str:
        if (
            self.speaker_baseline_mode == "neutral"
            and speaker_id not in self.neutral_fallback_speakers
        ):
            return "neutral baseline"

        return "speaker-specific enrollment average"

    def _build_prompt_for_sample(self, idx: int) -> str:
        real_idx = self.sample_indices[idx]

        if self.use_feature_prompt:
            if (
                self.disable_input_cue_text
                and self.prompt_type == "speaker_feature_answer_generation"
            ):
                return get_prompt("answer_generation", labels=self.label_names)

            features = self.acoustic_feature_cache[real_idx]

            if self.prompt_type in (
                "speaker_feature_answer_generation",
                "speaker_feature_answer_caption_generation",
                "speaker_feature_answer_evidence_generation",
            ):
                speaker_id = extract_speaker_id(self.all_file_paths[real_idx])
                baseline = self.speaker_baselines.get(
                    speaker_id,
                    compute_feature_baseline([features]),
                )
                feature_text = acoustic_features_to_speaker_relative_cues(
                    features,
                    baseline,
                    baseline_label=self._baseline_label_for_speaker(speaker_id),
                )
            else:
                feature_text = acoustic_features_to_text(features)

            return get_prompt(
                self.prompt_type,
                features=feature_text,
                labels=self.label_names,
            )

        return get_prompt(self.prompt_type, labels=self.label_names)

    def _build_target_for_sample(self, real_idx: int, label_text: str) -> str:
        if not self.use_reasoning_target:
            if self.use_acoustic_cue_target:
                features = self.acoustic_feature_cache[real_idx]
                speaker_id = extract_speaker_id(self.all_file_paths[real_idx])
                baseline = self.speaker_baselines.get(
                    speaker_id,
                    compute_feature_baseline([features]),
                )
                cue_categories = self._speaker_relative_cue_categories(
                    features,
                    baseline,
                )
                return self._format_acoustic_cue_target(cue_categories)

            if self.use_caption_target:
                features = self.acoustic_feature_cache[real_idx]
                speaker_id = extract_speaker_id(self.all_file_paths[real_idx])
                baseline = self.speaker_baselines.get(
                    speaker_id,
                    compute_feature_baseline([features]),
                )
                caption = acoustic_features_to_speaker_relative_cues(
                    features,
                    baseline,
                    baseline_label=self._baseline_label_for_speaker(speaker_id),
                )
                return f"<answer>{label_text}</answer><caption>{caption}</caption>"

            if self.use_evidence_target:
                features = self.acoustic_feature_cache[real_idx]
                speaker_id = extract_speaker_id(self.all_file_paths[real_idx])
                baseline = self.speaker_baselines.get(
                    speaker_id,
                    compute_feature_baseline([features]),
                )
                evidence = speaker_relative_evidence_sentence(
                    features,
                    baseline,
                    label_text,
                    baseline_label=self._baseline_label_for_speaker(speaker_id),
                )
                return f"<answer>{label_text}</answer><evidence>{evidence}</evidence>"

            if self.use_answer_tag_target:
                return f"<answer>{label_text}</answer>"
            return " " + label_text

        features = self.acoustic_feature_cache[real_idx]

        if self.use_speaker_reasoning:
            speaker_id = extract_speaker_id(self.all_file_paths[real_idx])
            baseline = self.speaker_baselines.get(
                speaker_id,
                compute_feature_baseline([features]),
            )
            caption = acoustic_features_to_speaker_relative_caption(
                features,
                baseline,
                baseline_label=self._baseline_label_for_speaker(speaker_id),
            )
        else:
            caption = acoustic_features_to_global_caption(features)

        reasoning = emotion_reasoning_sentence(
            label_text,
            baseline_label=(
                self._baseline_label_for_speaker(speaker_id)
                if self.use_speaker_reasoning
                else None
            ),
        )

        if self.prompt_type == "speaker_reasoning_generation_answer_first":
            return f"<answer>{label_text}</answer><think>{caption} {reasoning}</think>"

        return f"<think>{caption} {reasoning}</think><answer>{label_text}</answer>"

    def build_acoustic_cue_target_for_sample(self, idx: int) -> dict[str, str]:
        real_idx = self.sample_indices[idx]
        features = self.acoustic_feature_cache[real_idx]
        speaker_id = extract_speaker_id(self.all_file_paths[real_idx])
        baseline = self.speaker_baselines.get(
            speaker_id,
            compute_feature_baseline([features]),
        )
        return self._speaker_relative_cue_categories(features, baseline)

    def _relative_category(
        self,
        features,
        baseline,
        key: str,
        lower: str,
        similar: str,
        higher: str,
        threshold: float = 0.5,
    ) -> str:
        stats = baseline.get(key, {"mean": 0.0, "std": 1.0})
        std = float(stats.get("std", 1.0))
        if std < 1e-6:
            return similar

        z = (
            float(features.get(key, 0.0))
            - float(stats.get("mean", 0.0))
        ) / std

        if z > threshold:
            return higher
        if z < -threshold:
            return lower
        return similar

    def _speaker_relative_cue_categories(self, features, baseline) -> dict[str, str]:
        return {
            "pitch": self._relative_category(
                features,
                baseline,
                "pitch_mean",
                lower="lower",
                similar="similar",
                higher="higher",
            ),
            "energy": self._relative_category(
                features,
                baseline,
                "energy_mean",
                lower="lower",
                similar="similar",
                higher="higher",
            ),
            "rhythm": self._relative_category(
                features,
                baseline,
                "tempo",
                lower="slower",
                similar="similar",
                higher="faster",
            ),
            "duration": self._relative_category(
                features,
                baseline,
                "duration",
                lower="shorter",
                similar="similar",
                higher="longer",
            ),
        }

    def _format_acoustic_cue_target(self, cue_categories: dict[str, str]) -> str:
        return (
            "<caption>"
            f"<pitch>{cue_categories['pitch']}</pitch>"
            f"<energy>{cue_categories['energy']}</energy>"
            f"<rhythm>{cue_categories['rhythm']}</rhythm>"
            f"<duration>{cue_categories['duration']}</duration>"
            "</caption>"
        )

    def build_target_for_sample(self, idx: int, label_text: str) -> str:
        real_idx = self.sample_indices[idx]
        return self._build_target_for_sample(real_idx, label_text)

    def _build_generation_sample(self, idx: int):
        real_idx = self.sample_indices[idx]
        prompt = self._build_prompt_for_sample(idx)
        label_text = self._label_to_text(self.labels[real_idx])

        target = self._build_target_for_sample(real_idx, label_text)
        full_text = prompt + target

        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        full_encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = full_encoded["input_ids"].squeeze(0)
        lm_labels = input_ids.clone()

        prompt_len = prompt_encoded["input_ids"].shape[1]

        lm_labels[:prompt_len] = -100
        lm_labels[input_ids == self.tokenizer.pad_token_id] = -100

        loss_weights = None

        if self.prompt_type == "speaker_feature_answer_evidence_generation":
            loss_weights = self._build_answer_evidence_loss_weights(
                prompt_len=prompt_len,
                target=target,
                input_ids=input_ids,
            )

        answer_loss_mask = None
        if "<answer>" in target and "</answer>" in target:
            answer_loss_mask = self._build_answer_loss_mask(
                prompt_len=prompt_len,
                target=target,
                input_ids=input_ids,
            )

        return input_ids, lm_labels, loss_weights, answer_loss_mask

    def _encode_target_piece(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_answer_loss_mask(
        self,
        prompt_len: int,
        target: str,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mark answer-label target tokens for optional class-weighted loss.

        The mask covers the content between <answer> and </answer>, not the
        structural tags. Prompt and padding positions stay inactive.
        """
        answer_content_start = target.index("<answer>") + len("<answer>")
        answer_content_end = target.index("</answer>")

        prefix_len = len(self._encode_target_piece(target[:answer_content_start]))
        answer_len = len(
            self._encode_target_piece(
                target[answer_content_start:answer_content_end]
            )
        )

        answer_loss_mask = torch.zeros_like(input_ids, dtype=torch.float)
        available_target_len = max(0, input_ids.numel() - prompt_len)
        answer_start = prompt_len + prefix_len
        answer_end = min(answer_start + answer_len, prompt_len + available_target_len)

        if answer_start < answer_end:
            answer_loss_mask[answer_start:answer_end] = 1.0

        answer_loss_mask[input_ids == self.tokenizer.pad_token_id] = 0.0

        return answer_loss_mask

    def _build_answer_evidence_loss_weights(
        self,
        prompt_len: int,
        target: str,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build target-token weights for answer+evidence generation.

        The full answer span, including answer tags, receives a high weight so
        the model keeps learning the classification answer. The evidence span,
        including evidence tags, receives a low weight so it remains trainable
        without dominating the LM objective.
        """
        answer_start = target.index("<answer>")
        answer_end = target.index("</answer>") + len("</answer>")
        evidence_start = target.index("<evidence>")
        evidence_end = target.index("</evidence>") + len("</evidence>")

        target_token_weights = []
        cursor = 0

        for span_start, span_end, span_weight in (
            (answer_start, answer_end, self.answer_loss_weight),
            (evidence_start, evidence_end, self.evidence_loss_weight),
        ):
            if cursor < span_start:
                target_token_weights.extend(
                    [1.0] * len(self._encode_target_piece(target[cursor:span_start]))
                )

            target_token_weights.extend(
                [span_weight] * len(self._encode_target_piece(target[span_start:span_end]))
            )
            cursor = span_end

        if cursor < len(target):
            target_token_weights.extend(
                [1.0] * len(self._encode_target_piece(target[cursor:]))
            )

        loss_weights = torch.ones_like(input_ids, dtype=torch.float)
        available_target_len = max(0, input_ids.numel() - prompt_len)
        target_weight_len = min(len(target_token_weights), available_target_len)

        if target_weight_len > 0:
            loss_weights[prompt_len:prompt_len + target_weight_len] = torch.tensor(
                target_token_weights[:target_weight_len],
                dtype=torch.float,
            )

        loss_weights[input_ids == self.tokenizer.pad_token_id] = 0.0

        return loss_weights

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        real_idx = self.sample_indices[idx]

        item = {
            "input_ids": self.input_ids_list[idx],
            "labels": self.lm_labels_list[idx],
            "audio": self.embeddings[real_idx],
            "class_label": self.class_labels_list[idx],
        }

        if self.loss_weights_list[idx] is not None:
            item["loss_weights"] = self.loss_weights_list[idx]

        if self.answer_loss_masks_list[idx] is not None:
            item["answer_loss_mask"] = self.answer_loss_masks_list[idx]

        return item
