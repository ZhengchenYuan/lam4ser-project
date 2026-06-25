"""Extract encoder embeddings from the FAU AIBO corpus.

This uses the IS2009 Emotion Challenge 5-class labels and saves embeddings in
the same .pt format as the EMoDB preprocessing script.

Usage:
    python models/audio_encoder/preprocessing_aibo.py --encoder wavlm-large
    python models/audio_encoder/preprocessing_aibo.py --encoder wav2vec2-large-emotion
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Type

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import audiofile
import numpy as np
import torch
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
)

from data.dataset_configs import AIBO_LABELS


SAMPLING_RATE = 16000
MAX_DURATION_SEC = 5.0
MAX_SAMPLES = int(MAX_DURATION_SEC * SAMPLING_RATE)
EMBEDDINGS_DIR = "embeddings"

DATASET_DIR = os.environ.get("AIBO_DATA_DIR", "dataset")
WAV_DIR = os.path.join(DATASET_DIR, "wav")
LABELS_FILE = os.path.join(
    DATASET_DIR,
    "labels",
    "IS2009EmotionChallenge",
    "chunk_labels_5cl_corpus.txt",
)

AIBO_LABEL_MAP = {
    "A": "anger",
    "E": "emphatic",
    "N": "neutral",
    "P": "positive",
    "R": "rest",
}


@dataclass
class EncoderSpec:
    model_id: str
    hidden_dim: int
    processor_cls: Type
    model_cls: Type


ENCODERS: dict = {
    "wav2vec2-base": EncoderSpec(
        model_id="facebook/wav2vec2-base-960h",
        hidden_dim=768,
        processor_cls=Wav2Vec2Processor,
        model_cls=Wav2Vec2Model,
    ),
    "wav2vec2-large-emotion": EncoderSpec(
        model_id="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        hidden_dim=1024,
        processor_cls=Wav2Vec2Processor,
        model_cls=Wav2Vec2Model,
    ),
    "wavlm-large": EncoderSpec(
        model_id="microsoft/wavlm-large",
        hidden_dim=1024,
        processor_cls=Wav2Vec2FeatureExtractor,
        model_cls=WavLMModel,
    ),
    "hubert-large": EncoderSpec(
        model_id="facebook/hubert-large-ls960-ft",
        hidden_dim=1024,
        processor_cls=Wav2Vec2FeatureExtractor,
        model_cls=HubertModel,
    ),
}


def _load_aibo_index() -> list[tuple[str, str]]:
    """Read the IS2009 5-class label file as (wav_path, label_name)."""
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"AIBO label file not found: {LABELS_FILE}")

    index = []
    with open(LABELS_FILE, encoding="utf-8") as fp:
        for line in fp:
            parts = line.split()
            if not parts:
                continue

            chunk_name, code = parts[0], parts[1]
            if code not in AIBO_LABEL_MAP:
                raise ValueError(f"Unknown AIBO label code {code!r} for {chunk_name}")

            wav_path = os.path.join(WAV_DIR, f"{chunk_name}.wav")
            index.append((wav_path, AIBO_LABEL_MAP[code]))

    return index


def extract(
    encoder_name: str,
    output_path: str | None = None,
    limit: int | None = None,
) -> str:
    spec = ENCODERS[encoder_name]

    if output_path is None:
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        output_path = os.path.join(
            EMBEDDINGS_DIR,
            f"aibo_{encoder_name}_embeddings.pt",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoder : {encoder_name} ({spec.model_id})")
    print(f"Device  : {device}")
    print(f"Output  : {output_path}")

    print("\nLoading AIBO (IS2009 Emotion Challenge, 5-class)...")
    index = _load_aibo_index()
    if limit is not None:
        index = index[:limit]
    print(f"Samples : {len(index)}")

    label2idx = {label: idx for idx, label in enumerate(AIBO_LABELS)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    print(f"\nLoading {spec.model_id}...")
    processor = spec.processor_cls.from_pretrained(spec.model_id)
    model = spec.model_cls.from_pretrained(spec.model_id).to(device).eval()

    embeddings, labels, file_paths = [], [], []

    for i, (file_path, emotion) in enumerate(index):
        label_int = label2idx[emotion]

        signal, _ = audiofile.read(file_path, always_2d=False)
        if len(signal) > MAX_SAMPLES:
            signal = signal[:MAX_SAMPLES]
        else:
            signal = np.pad(signal, (0, MAX_SAMPLES - len(signal)), mode="constant")

        inputs = processor(
            signal,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            hidden = model(input_values).last_hidden_state.squeeze(0).cpu()

        embeddings.append(hidden)
        labels.append(label_int)
        file_paths.append(file_path)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(index)}")

    T_audio = embeddings[0].shape[0]
    print(
        f"\nDone. T_audio={T_audio}, hidden_dim={spec.hidden_dim}, "
        f"samples={len(embeddings)}"
    )

    torch.save(
        {
            "dataset": "aibo",
            "embeddings": embeddings,
            "labels": labels,
            "file_paths": file_paths,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "T_audio": T_audio,
            "hidden_dim": spec.hidden_dim,
            "encoder": encoder_name,
            "model_id": spec.model_id,
        },
        output_path,
    )
    print(f"Saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio encoder embeddings from AIBO."
    )
    parser.add_argument(
        "--encoder",
        choices=list(ENCODERS),
        default="wav2vec2-base",
        help=(
            "Encoder to use. "
            f"Options: {', '.join(ENCODERS)}"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt path (default: embeddings/aibo_<encoder>_embeddings.pt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples for smoke-testing.",
    )
    args = parser.parse_args()
    extract(args.encoder, args.output, args.limit)
