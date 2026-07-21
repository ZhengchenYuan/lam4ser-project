#!/usr/bin/env python3
"""Generate AIBO teacher rationales with Qwen3-Omni audio understanding."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import extract_speaker_id, speaker_independent_split
from data.dataset_configs import get_dataset_config


DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PROMPT_VERSION = "audioqwen3_teacher_rationale_v1"
TEACHER_PROMPT = """Analyze the emotional speech in the audio.
Describe only audible acoustic evidence, such as pitch, loudness/energy, rhythm/speaking rate, duration, voice quality, and arousal.
Do not use transcript content unless it is clearly audible.
Do not mention the ground-truth emotion label.
Do not say "these cues support [emotion]".
Return a concise rationale in 2-4 sentences."""

AIBO_LABEL_MAP = {
    "A": "anger",
    "E": "emphatic",
    "N": "neutral",
    "P": "positive",
    "R": "rest",
}


@dataclass(frozen=True)
class AIBOSample:
    sample_id: str
    split: str
    answer: str
    audio_path: str


class _SplitDataset:
    """Small adapter for the repository's shared speaker split function."""

    def __init__(self, samples: list[tuple[str, str]]):
        self.speaker_ids = [extract_speaker_id(path) for path, _ in samples]

    def __len__(self) -> int:
        return len(self.speaker_ids)


def _load_aibo_index() -> list[tuple[str, str]]:
    """Read the same IS2009 five-class index used by AIBO preprocessing."""
    dataset_dir = Path(os.environ.get("AIBO_DATA_DIR", "dataset"))
    labels_path = (
        dataset_dir
        / "labels"
        / "IS2009EmotionChallenge"
        / "chunk_labels_5cl_corpus.txt"
    )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"AIBO label file not found: {labels_path}. Set AIBO_DATA_DIR to "
            "the corpus root if it is not under ./dataset."
        )

    samples: list[tuple[str, str]] = []
    with labels_path.open(encoding="utf-8") as labels_file:
        for line_number, line in enumerate(labels_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2 or fields[1] not in AIBO_LABEL_MAP:
                raise ValueError(
                    f"Invalid AIBO label at {labels_path}:{line_number}: {line.strip()}"
                )
            sample_id, label_code = fields[:2]
            audio_path = dataset_dir / "wav" / f"{sample_id}.wav"
            samples.append((str(audio_path), AIBO_LABEL_MAP[label_code]))

    if not samples:
        raise ValueError(f"No AIBO samples found in {labels_path}.")
    return samples


def _without_neutral_enrollment(
    samples: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Match the neutral enrollment exclusion used by rationale generation."""
    by_speaker_label: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (audio_path, answer) in enumerate(samples):
        by_speaker_label[(extract_speaker_id(audio_path), answer)].append(index)
    for indices in by_speaker_label.values():
        indices.sort(key=lambda index: Path(samples[index][0]).name)

    labels_by_speaker: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for (speaker_id, answer), indices in by_speaker_label.items():
        labels_by_speaker[speaker_id][answer] = indices

    enrollment_indices: set[int] = set()
    for labels in labels_by_speaker.values():
        if labels.get("neutral"):
            enrollment_indices.add(labels["neutral"][0])
        else:
            # This is the same fallback as
            # EmoDBGenerationDataset._select_neutral_enrollment.
            enrollment_indices.update(indices[0] for indices in labels.values())

    return [
        sample for index, sample in enumerate(samples) if index not in enrollment_indices
    ]


def load_aibo_samples() -> list[AIBOSample]:
    """Build sample IDs, answers, and splits compatible with generation data."""
    indexed_samples = _without_neutral_enrollment(_load_aibo_index())
    config = get_dataset_config("aibo")
    train_indices, val_indices, test_indices = speaker_independent_split(
        _SplitDataset(indexed_samples),
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
    )
    split_by_index = {
        **{index: "train" for index in train_indices},
        # The existing AIBO rationale JSONL calls this split "validation".
        **{index: "validation" for index in val_indices},
        **{index: "test" for index in test_indices},
    }
    return [
        AIBOSample(
            sample_id=Path(audio_path).stem,
            split=split_by_index[index],
            answer=answer,
            audio_path=audio_path,
        )
        for index, (audio_path, answer) in enumerate(indexed_samples)
    ]


def _resolve_dtype(dtype_name: str, torch_module: Any) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
    }[dtype_name]


def load_audioqwen3_model(
    model_id: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    """Load Qwen3-Omni using the official Transformers text-output path."""
    try:
        import torch
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-Omni requires transformers>=5.2.0, accelerate, torch, "
            "qwen-omni-utils, and ffmpeg."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    # Current Transformers exposes the native Qwen3-Omni classes. Disabling
    # audio output omits the Talker because this pipeline only needs text.
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        dtype=_resolve_dtype(dtype, torch),
        device_map=device if device == "auto" else {"": device},
        trust_remote_code=trust_remote_code,
        enable_audio_output=False,
        low_cpu_mem_usage=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, processor


def generate_audioqwen3_rationale(
    model: Any,
    processor: Any,
    audio_path: str,
    prompt: str,
    generation_config: dict[str, Any],
) -> str:
    """Generate one text rationale from a raw audio file and instruction."""
    import torch
    from qwen_omni_utils import process_mm_info

    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(
        messages,
        use_audio_in_video=False,
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    input_length = inputs["input_ids"].shape[1]

    # Qwen3-Omni routes text generation through its Thinker, so generation
    # controls use the official thinker_* keyword names.
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=False,
            thinker_max_new_tokens=generation_config["max_new_tokens"],
            thinker_do_sample=generation_config["temperature"] > 0,
            thinker_temperature=generation_config["temperature"],
            thinker_top_p=generation_config["top_p"],
        )

    # Some supported Transformers revisions return (text_ids, audio), while
    # text-only revisions return text_ids directly.
    text_ids = generated[0] if isinstance(generated, tuple) else generated
    sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
    rationale = processor.batch_decode(
        sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    del generated, text_ids, sequences, inputs, audios, images, videos
    if not rationale:
        raise RuntimeError("Qwen3-Omni returned an empty rationale.")
    return rationale


def _completed_sample_ids(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open(encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed.add(str(record["sample_id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid resume record at {output_path}:{line_number}"
                ) from exc
    return completed


def _record_for_sample(
    sample: AIBOSample,
    model_id: str,
    rationale: str,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "answer": sample.answer,
        "audio_path": sample.audio_path,
        "teacher_model": model_id,
        "prompt_version": PROMPT_VERSION,
        "teacher_prompt": TEACHER_PROMPT,
        "rationale": rationale,
        "status": status,
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aibo",), default="aibo")
    parser.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="all"
    )
    parser.add_argument("--max_samples", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations/aibo_audioqwen3_teacher_rationales_pilot.jsonl"),
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "auto"), default="auto"
    )
    parser.add_argument(
        "--prompt_version", choices=(PROMPT_VERSION,), default=PROMPT_VERSION
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max_samples must be a positive integer")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be a positive integer")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    split_name = "validation" if args.split == "val" else args.split
    samples = load_aibo_samples()
    if split_name != "all":
        samples = [sample for sample in samples if sample.split == split_name]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    completed = _completed_sample_ids(args.output) if args.resume else set()
    samples = [sample for sample in samples if sample.sample_id not in completed]

    print(f"Selected {len(samples)} AIBO samples ({len(completed)} skipped).")
    if not samples:
        print("Nothing to generate.")
        return

    model, processor = load_audioqwen3_model(
        args.model_id,
        args.device,
        args.dtype,
        args.trust_remote_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    ok_count = 0
    failed_count = 0
    with args.output.open(mode, encoding="utf-8") as output_file:
        for index, sample in enumerate(samples, start=1):
            try:
                rationale = generate_audioqwen3_rationale(
                    model,
                    processor,
                    sample.audio_path,
                    TEACHER_PROMPT,
                    generation_config,
                )
                record = _record_for_sample(
                    sample, args.model_id, rationale, "ok", None
                )
                ok_count += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                record = _record_for_sample(
                    sample, args.model_id, "", "failed", error
                )
                failed_count += 1

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            if index % 10 == 0 or index == len(samples):
                print(
                    f"Progress: {index}/{len(samples)} "
                    f"(ok={ok_count}, failed={failed_count})"
                )

    print(
        f"Wrote {len(samples)} records to {args.output} "
        f"(ok={ok_count}, failed={failed_count})."
    )


if __name__ == "__main__":
    main()
