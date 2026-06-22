import os
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from data.prompts import get_prompt
from features.acoustic_features import extract_acoustic_features
from features.feature_prompt import (
    acoustic_features_to_global_caption,
    acoustic_features_to_speaker_relative_caption,
    acoustic_features_to_text,
    compute_feature_baseline,
    emotion_reasoning_sentence,
)


GENERATION_PROMPT_TYPES = (
    "generation",
    "feature_generation",
    "reasoning_generation_global",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
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


def extract_speaker_id(file_path: str) -> str:
    """
    EmoDB filenames start with a two-digit speaker ID, e.g. 03a01Fa.wav.
    """
    basename = os.path.basename(file_path)
    if len(basename) < 2:
        return "unknown"
    return basename[:2]


class EmoDBGenerationDataset(Dataset):
    def __init__(
        self,
        embeddings_path: str,
        prompt_type: str = "generation",
        max_length: int = 96,
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
        self.use_feature_prompt = "feature" in prompt_type
        self.use_reasoning_target = prompt_type in REASONING_PROMPT_TYPES
        self.use_speaker_reasoning = prompt_type in SPEAKER_REASONING_PROMPT_TYPES

        data = torch.load(embeddings_path, weights_only=False)

        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.label2idx = data["label2idx"]
        self.idx2label = data["idx2label"]

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

        if self.use_feature_prompt and self.file_paths is None:
            raise ValueError(
                "feature_generation requires wav file paths, but no key among "
                "('file_paths', 'paths', 'files') was found in the embeddings file."
            )

        if self.use_reasoning_target and self.file_paths is None:
            raise ValueError(
                f"{prompt_type} requires wav file paths for acoustic caption targets, "
                "but no key among ('file_paths', 'paths', 'files') was found in the "
                "embeddings file."
            )

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.acoustic_feature_cache = None
        self.speaker_baselines = {}
        self.enrollment_indices = set()
        self.sample_indices = list(range(len(self.embeddings)))

        if self.use_feature_prompt or self.use_reasoning_target:
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

        if self.use_speaker_reasoning:
            self._build_enrollment_sets()

        self.input_ids_list = []
        self.lm_labels_list = []
        self.class_labels_list = []

        for idx in range(len(self.sample_indices)):
            input_ids, lm_labels = self._build_generation_sample(idx)
            self.input_ids_list.append(input_ids)
            self.lm_labels_list.append(lm_labels)
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
            self.enrollment_indices.add(indices[0])

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
        print(f"  Speakers:           {len(enrollment_by_speaker)}")
        print(f"  Enrollment samples: {len(self.enrollment_indices)}")
        print(f"  Target samples:     {len(self.sample_indices)}")
        fallback_messages = 0

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

    def _build_prompt_for_sample(self, idx: int) -> str:
        real_idx = self.sample_indices[idx]

        if self.use_feature_prompt:
            features = self.acoustic_feature_cache[real_idx]
            feature_text = acoustic_features_to_text(features)
            return get_prompt(self.prompt_type, features=feature_text)

        return get_prompt(self.prompt_type)

    def _build_target_for_sample(self, real_idx: int, label_text: str) -> str:
        if not self.use_reasoning_target:
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
            )
        else:
            caption = acoustic_features_to_global_caption(features)

        reasoning = emotion_reasoning_sentence(label_text)

        if self.prompt_type == "speaker_reasoning_generation_answer_first":
            return f"<answer>{label_text}</answer><think>{caption} {reasoning}</think>"

        return f"<think>{caption} {reasoning}</think><answer>{label_text}</answer>"

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

        return input_ids, lm_labels

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        real_idx = self.sample_indices[idx]

        return {
            "input_ids": self.input_ids_list[idx],
            "labels": self.lm_labels_list[idx],
            "audio": self.embeddings[real_idx],
            "class_label": self.class_labels_list[idx],
        }
