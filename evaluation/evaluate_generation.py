import argparse
import csv
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data.dataset_configs import DATASET_CONFIGS, get_dataset_config
from data.generation_dataset import EmoDBGenerationDataset, SPEAKER_BASELINE_PROMPT_TYPES
from data.dataset import speaker_independent_split
from data.prompts import LABELS as DEFAULT_LABELS
from data.tokenizer_utils import build_generation_tokenizer
from models.compression.compressor import AudioCompressor
from models.audio_gpt2_generation import AudioGPT2Generation


GENERATION_PROMPT_TYPES = [
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
]

REASONING_PROMPT_TYPES = {
    "reasoning_generation_global",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
}

ANSWER_TAG_PROMPT_TYPES = REASONING_PROMPT_TYPES | {
    "answer_generation",
    "speaker_feature_answer_generation",
    "speaker_feature_answer_caption_generation",
    "speaker_feature_answer_evidence_generation",
}

LABEL_CONSTRAINED_PROMPT_TYPES = {
    "answer_generation",
    "speaker_feature_answer_generation",
    "speaker_feature_answer_caption_generation",
    "speaker_feature_answer_evidence_generation",
}

ACOUSTIC_CUE_PROMPT_TYPE = "speaker_acoustic_cue_generation"
CUE_NAMES = ("pitch", "energy", "rhythm", "duration")


def _checkpoint_tag(
    encoder: str,
    prompt_type: str,
    speaker_baseline_mode: str,
    class_weighted_answer_loss: bool = False,
    class_weight_mode: str = "inverse",
    class_weight_power: float = 1.0,
    class_weight_max: float = 5.0,
) -> str:
    tag = f"{encoder}_{prompt_type}"

    if prompt_type in SPEAKER_BASELINE_PROMPT_TYPES:
        tag += f"_{speaker_baseline_mode}"

    if class_weighted_answer_loss:
        max_tag = (
            str(float(class_weight_max))
            if class_weight_max is not None and class_weight_max > 0
            else "none"
        )
        tag += (
            f"_weighted_{class_weight_mode}"
            f"_p{float(class_weight_power)}"
            f"_m{max_tag}"
        )

    return f"{tag}_generation"


def _max_length_for_prompt_type(prompt_type: str) -> int:
    if prompt_type == ACOUSTIC_CUE_PROMPT_TYPE:
        return 128
    if "reasoning_generation" in prompt_type:
        return 224
    if "feature" in prompt_type:
        return 128
    return 96


def _max_new_tokens_for_prompt_type(prompt_type: str) -> int:
    if prompt_type == ACOUSTIC_CUE_PROMPT_TYPE:
        return 32
    if prompt_type in REASONING_PROMPT_TYPES:
        return 96
    return 5


def _build_config(
    encoder: str,
    prompt_type: str,
    dataset: str = "emodb",
    checkpoint_path: str | None = None,
    candidate_scoring: str = "none",
    generate_evidence: bool = False,
    no_audio: bool = False,
    cue_perturbation: str = "none",
    speaker_baseline_mode: str = "neutral",
    class_weighted_answer_loss: bool = False,
    class_weight_mode: str = "inverse",
    class_weight_power: float = 1.0,
    class_weight_max: float = 5.0,
) -> dict:
    dataset_config = get_dataset_config(dataset)
    tag = _checkpoint_tag(
        encoder=encoder,
        prompt_type=prompt_type,
        speaker_baseline_mode=speaker_baseline_mode,
        class_weighted_answer_loss=class_weighted_answer_loss,
        class_weight_mode=class_weight_mode,
        class_weight_power=class_weight_power,
        class_weight_max=class_weight_max,
    )
    max_new_tokens = _max_new_tokens_for_prompt_type(prompt_type)

    if generate_evidence:
        if prompt_type != "speaker_feature_answer_evidence_generation":
            raise ValueError(
                "--generate_evidence is only supported with "
                "speaker_feature_answer_evidence_generation."
            )
        max_new_tokens = 64

    return {
        "dataset": dataset,
        "encoder": encoder,
        "prompt_type": prompt_type,
        "max_prompt_length": _max_length_for_prompt_type(prompt_type),
        "embeddings_path": (
            f"embeddings/{dataset_config['embeddings_prefix']}"
            f"{encoder}_embeddings.pt"
        ),
        "batch_size": 1,
        "adapter_dim": 64,
        "dropout": 0.3,
        "target_audio_len": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_speakers": dataset_config["val_speakers"],
        "test_speakers": dataset_config["test_speakers"],
        "checkpoint_path": (
            checkpoint_path
            or f"{dataset_config['checkpoint_dir']}/{tag}_best.pt"
        ),
        "max_new_tokens": max_new_tokens,
        "candidate_scoring": candidate_scoring,
        "generate_evidence": generate_evidence,
        "no_audio": no_audio,
        "cue_perturbation": cue_perturbation,
        "speaker_baseline_mode": speaker_baseline_mode,
        "class_weighted_answer_loss": class_weighted_answer_loss,
        "class_weight_mode": class_weight_mode,
        "class_weight_power": class_weight_power,
        "class_weight_max": class_weight_max,
        "preprocessing_script": dataset_config["preprocessing_script"],
    }


def normalize_generated_label(text: str, labels: list[str]) -> str | None:
    """
    Map generated text back to one of the emotion labels.

    Example:
        " anger" -> "anger"
        "The emotion is anger." -> "anger"
        "happy" -> None, because the label is "happiness"
    """
    text = text.lower().strip()

    for label in labels:
        if label in text:
            return label

    return None


def extract_reasoning_blocks(text: str) -> tuple[str, str, bool]:
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)

    think_text = think_match.group(1).strip() if think_match else ""
    answer_text = answer_match.group(1).strip() if answer_match else ""
    format_valid = bool(think_text and answer_text)

    return think_text, answer_text, format_valid


def extract_evidence_text(text: str) -> tuple[str, bool]:
    evidence_match = re.search(
        r"<evidence>(.*?)</evidence>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    evidence_text = evidence_match.group(1).strip() if evidence_match else ""
    format_valid = bool(
        re.search(r"<evidence>", text, flags=re.IGNORECASE)
        and re.search(r"</evidence>", text, flags=re.IGNORECASE)
    )

    return evidence_text, format_valid


EVIDENCE_CUE_WORDS = {
    "energy",
    "pitch",
    "rhythm",
    "tempo",
    "duration",
    "activation",
    "arousal",
    "tense",
    "subdued",
    "expressive",
}


def mentions_evidence_cue(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in EVIDENCE_CUE_WORDS)


def extract_speaker_relative_cues(prompt: str) -> str:
    match = re.search(
        r"Speaker-relative acoustic cues:\s*(.*?)(?:\.\s*)?$",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def replace_speaker_relative_cues(prompt: str, cue_text: str) -> str:
    return re.sub(
        r"(Speaker-relative acoustic cues:\s*)(.*?)(\.\s*)?$",
        rf"\g<1>{cue_text}\g<3>",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )


def invert_cue_text(cue_text: str) -> str:
    replacements = (
        ("higher", "__CUE_HIGHER__"),
        ("lower", "higher"),
        ("__CUE_HIGHER__", "lower"),
        ("faster", "__CUE_FASTER__"),
        ("slower", "faster"),
        ("__CUE_FASTER__", "slower"),
        ("longer", "__CUE_LONGER__"),
        ("shorter", "longer"),
        ("__CUE_LONGER__", "shorter"),
        ("high", "__CUE_HIGH__"),
        ("low", "high"),
        ("__CUE_HIGH__", "low"),
    )

    perturbed = cue_text
    for source, target in replacements:
        perturbed = re.sub(rf"\b{source}\b", target, perturbed, flags=re.IGNORECASE)

    return perturbed


def build_eval_prompt_overrides(dataset, test_idx, cue_perturbation: str):
    if cue_perturbation == "none":
        return {}

    prompts = {
        idx: build_prompt_for_eval(dataset, idx)
        for idx in test_idx
    }

    if cue_perturbation == "invert":
        return {
            idx: replace_speaker_relative_cues(
                prompt,
                invert_cue_text(extract_speaker_relative_cues(prompt)),
            )
            for idx, prompt in prompts.items()
        }

    if cue_perturbation == "shuffle":
        cue_texts = [
            extract_speaker_relative_cues(prompts[idx])
            for idx in test_idx
        ]
        if not cue_texts:
            return prompts

        shifted_cues = cue_texts[1:] + cue_texts[:1]
        return {
            idx: replace_speaker_relative_cues(prompts[idx], shifted_cue)
            for idx, shifted_cue in zip(test_idx, shifted_cues)
        }

    raise ValueError(f"Unknown cue perturbation: {cue_perturbation}")


def cue_faithfulness_ok(cue_text: str, evidence_text: str) -> bool:
    cue_text = cue_text.lower()
    evidence_text = evidence_text.lower()
    contradiction_pairs = (
        ("higher energy", "lower energy"),
        ("lower energy", "higher energy"),
        ("higher pitch", "lower pitch"),
        ("lower pitch", "higher pitch"),
        ("faster rhythm", "slower rhythm"),
        ("slower rhythm", "faster rhythm"),
        ("longer duration", "shorter duration"),
        ("shorter duration", "longer duration"),
    )

    for cue_phrase, contradictory_phrase in contradiction_pairs:
        if cue_phrase in cue_text and contradictory_phrase in evidence_text:
            return False

    return True


def parse_acoustic_cues(text: str) -> tuple[dict[str, str], bool]:
    parsed = {}
    for cue_name in CUE_NAMES:
        match = re.search(
            rf"<{cue_name}>(.*?)</{cue_name}>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        parsed[cue_name] = match.group(1).strip().lower() if match else ""

    format_valid = bool(
        re.search(r"<caption>", text, flags=re.IGNORECASE)
        and re.search(r"</caption>", text, flags=re.IGNORECASE)
        and all(parsed[cue_name] for cue_name in CUE_NAMES)
    )

    return parsed, format_valid


def parse_generated_label(
    text: str,
    prompt_type: str,
    labels: list[str],
) -> tuple[str | None, str, str, bool]:
    think_text, answer_text, format_valid = extract_reasoning_blocks(text)

    if prompt_type in ANSWER_TAG_PROMPT_TYPES and answer_text:
        pred_label = normalize_generated_label(answer_text, labels)
    else:
        pred_label = normalize_generated_label(text, labels)

    if pred_label is None and answer_text:
        pred_label = normalize_generated_label(answer_text, labels)

    if prompt_type in LABEL_CONSTRAINED_PROMPT_TYPES:
        format_valid = bool(answer_text)

    return pred_label, think_text, answer_text, format_valid


def build_prompt_for_eval(dataset, idx: int) -> str:
    """
    Rebuild only the prompt part for generation.

    During training, the dataset input is:
        prompt + answer

    During evaluation, we should start from:
        prompt

    and let the model generate the answer.
    """
    return dataset._build_prompt_for_sample(idx)


def language_model_loss_per_candidate(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss = loss.view(shift_labels.shape)
    token_mask = shift_labels.ne(-100)

    return loss.sum().item(), token_mask.sum().item()


def load_generation_state_dict(model, state_dict):
    current_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            current_shape = current_state[key].shape if key in current_state else None
            skipped.append((key, tuple(value.shape), current_shape))

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)

    if skipped:
        print("Skipped checkpoint tensors with incompatible shapes:")
        for key, old_shape, new_shape in skipped:
            print(f"  {key}: checkpoint {old_shape}, model {new_shape}")

    if missing:
        print(f"Missing tensors after compatible load: {len(missing)}")
    if unexpected:
        print(f"Unexpected tensors after compatible load: {unexpected}")


def build_candidate_target(dataset, idx: int, label: str, mode: str) -> str:
    if mode == "answer":
        if dataset.prompt_type in ANSWER_TAG_PROMPT_TYPES:
            return f"<answer>{label}</answer>"
        return " " + label

    if mode == "full_target":
        return dataset.build_target_for_sample(idx, label)

    raise ValueError(f"Unknown candidate scoring mode: {mode}")


@torch.no_grad()
def evaluate_candidate_scoring(
    model,
    tokenizer,
    compressor,
    dataset,
    test_idx,
    loader,
    config,
    mode: str,
    label_names: list[str],
    prompt_overrides=None,
):
    y_true = []
    y_pred = []

    for local_batch_idx, batch in enumerate(loader):
        original_idx = test_idx[local_batch_idx]
        prompt = (
            prompt_overrides.get(original_idx)
            if prompt_overrides is not None and original_idx in prompt_overrides
            else build_prompt_for_eval(dataset, original_idx)
        )
        audio_compressed = None
        if not config["no_audio"]:
            audio = batch["audio"].to(config["device"])
            audio_compressed = compressor(audio)
        class_label = batch["class_label"].item()

        best_label = None
        best_score = float("inf")

        prompt_encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=config["max_prompt_length"],
            return_tensors="pt",
        )
        prompt_len = prompt_encoded["input_ids"].shape[1]

        for label in label_names:
            target = build_candidate_target(dataset, original_idx, label, mode)
            full_text = prompt + target

            encoded = tokenizer(
                full_text,
                max_length=config["max_prompt_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(config["device"])
            lm_labels = input_ids.clone()
            lm_labels[:, :prompt_len] = -100
            lm_labels[input_ids == tokenizer.pad_token_id] = -100

            logits = model(input_ids, audio_compressed)
            loss_sum, token_count = language_model_loss_per_candidate(
                logits,
                lm_labels,
            )
            score = loss_sum / max(token_count, 1)

            if score < best_score:
                best_score = score
                best_label = label

        y_true.append(dataset.idx2label[int(class_label)])
        y_pred.append(best_label or "invalid")

    print()
    print(f"Candidate-scoring results ({mode}):")
    _print_classification_metrics(y_true, y_pred, label_names)


def _print_classification_metrics(y_true, y_pred, label_names: list[str]):
    all_eval_labels = label_names + ["invalid"]

    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=all_eval_labels,
        average="weighted",
        zero_division=0,
    )
    invalid_count = sum(1 for pred in y_pred if pred == "invalid")
    validity = 1.0 - invalid_count / max(len(y_pred), 1)

    print(f"  Accuracy:                 {acc:.4f}")
    print(f"  Weighted F1:              {weighted_f1:.4f}")
    print(f"  Generated label validity: {validity:.4f}")
    print()

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=label_names,
        average=None,
        zero_division=0,
    )
    print("Per-class F1:")
    for label, score in zip(label_names, per_class_f1):
        print(f"  {label}: {score:.4f}")
    print()

    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=all_eval_labels,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(f"Labels: {all_eval_labels}")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=all_eval_labels,
        )
    )

    print()
    print("Prediction distribution:")
    print(Counter(y_pred))


def output_suffix(config) -> str:
    parts = []
    if config["no_audio"]:
        parts.append("no_audio")
    if config["cue_perturbation"] != "none":
        parts.append(f"cue_{config['cue_perturbation']}")
    if config["generate_evidence"]:
        parts.append("evidence")
    return "_" + "_".join(parts) if parts else ""


def write_generation_samples(config, sample_rows):
    os.makedirs("evaluation_outputs", exist_ok=True)
    sample_path = os.path.join(
        "evaluation_outputs",
        (
            f"{config['encoder']}_{config['prompt_type']}"
            f"{output_suffix(config)}_samples.csv"
        ),
    )
    fieldnames = [
        "prompt_type",
        "true_label",
        "parsed_prediction",
        "speaker_relative_cues",
        "generated_text",
        "think_text",
        "answer_text",
        "evidence_text",
        "evidence_format_valid",
        "evidence_non_empty",
        "evidence_cue_mention",
        "evidence_faithfulness_ok",
        "true_pitch",
        "pred_pitch",
        "true_energy",
        "pred_energy",
        "true_rhythm",
        "pred_rhythm",
        "true_duration",
        "pred_duration",
        "cue_format_valid",
        "cue_exact_match",
    ]

    with open(sample_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows[:50])

    return sample_path


def _single_token_id(tokenizer, token: str) -> int | None:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def _structured_token_ids(tokenizer):
    return {
        "think_start": _single_token_id(tokenizer, "<think>"),
        "think_end": _single_token_id(tokenizer, "</think>"),
        "caption_start": _single_token_id(tokenizer, "<caption>"),
        "caption_end": _single_token_id(tokenizer, "</caption>"),
        "answer_start": _single_token_id(tokenizer, "<answer>"),
        "answer_end": _single_token_id(tokenizer, "</answer>"),
        "evidence_start": _single_token_id(tokenizer, "<evidence>"),
        "evidence_end": _single_token_id(tokenizer, "</evidence>"),
    }


def _contains_token(input_ids, token_id: int | None) -> bool:
    if token_id is None:
        return False
    return bool(input_ids.eq(token_id).any().item())


def _label_token_sequences(tokenizer, label_names: list[str]):
    return {
        label: tokenizer.encode(label, add_special_tokens=False)
        for label in label_names
    }


def _answer_label_prefix(generated, answer_start_id: int | None):
    if answer_start_id is None:
        return []

    token_ids = generated[0].tolist()
    for idx in range(len(token_ids) - 1, -1, -1):
        if token_ids[idx] == answer_start_id:
            return token_ids[idx + 1:]

    return []


def _allowed_answer_generation_tokens(
    generated,
    structured_ids,
    label_sequences,
):
    answer_start_id = structured_ids["answer_start"]
    answer_end_id = structured_ids["answer_end"]

    if answer_start_id is None or answer_end_id is None:
        return None

    if _contains_token(generated, answer_end_id):
        return None

    if not _contains_token(generated, answer_start_id):
        return [answer_start_id]

    label_prefix = _answer_label_prefix(generated, answer_start_id)

    for sequence in label_sequences.values():
        if label_prefix == sequence:
            return [answer_end_id]

    allowed = set()
    for sequence in label_sequences.values():
        if sequence[:len(label_prefix)] == label_prefix:
            if len(sequence) > len(label_prefix):
                allowed.add(sequence[len(label_prefix)])

    return sorted(allowed)


def _mask_except(next_token_logits, allowed_token_ids):
    if allowed_token_ids is None:
        return

    masked_logits = next_token_logits.new_full(
        next_token_logits.shape,
        -float("inf"),
    )
    masked_logits[:, allowed_token_ids] = next_token_logits[:, allowed_token_ids]
    next_token_logits.copy_(masked_logits)


@torch.no_grad()
def greedy_generate(
    model,
    tokenizer,
    input_ids,
    audio_hidden,
    max_new_tokens: int = 5,
    prompt_type: str = "generation",
    generate_evidence: bool = False,
    label_names: list[str] | None = None,
):
    """
    Greedy decoding for a short emotion label.

    Args:
        input_ids:
            Prompt token ids, shape [1, T].

        audio_hidden:
            Compressed audio embeddings, shape [1, T_audio, audio_dim].

    Returns:
        Full generated token ids, including the prompt.
    """
    model.eval()

    generated = input_ids
    structured_ids = _structured_token_ids(tokenizer)
    label_sequences = _label_token_sequences(tokenizer, label_names or DEFAULT_LABELS)

    for _ in range(max_new_tokens):
        logits = model(generated, audio_hidden)
        next_token_logits = logits[:, -1, :]

        if prompt_type in LABEL_CONSTRAINED_PROMPT_TYPES:
            allowed_token_ids = _allowed_answer_generation_tokens(
                generated,
                structured_ids,
                label_sequences,
            )
            _mask_except(next_token_logits, allowed_token_ids)

        if (
            generate_evidence
            and structured_ids["evidence_start"] is not None
            and _contains_token(generated, structured_ids["answer_end"])
            and not _contains_token(generated, structured_ids["evidence_start"])
        ):
            _mask_except(next_token_logits, [structured_ids["evidence_start"]])

        if _contains_token(generated, structured_ids["answer_start"]):
            token_id = structured_ids["answer_start"]
            if token_id is not None:
                next_token_logits[:, token_id] = -float("inf")

        if _contains_token(generated, structured_ids["answer_end"]):
            token_id = structured_ids["answer_end"]
            if token_id is not None:
                next_token_logits[:, token_id] = -float("inf")

        if _contains_token(generated, structured_ids["evidence_start"]):
            token_id = structured_ids["evidence_start"]
            if token_id is not None:
                next_token_logits[:, token_id] = -float("inf")

        if _contains_token(generated, structured_ids["think_start"]):
            token_id = structured_ids["think_start"]
            if token_id is not None:
                next_token_logits[:, token_id] = -float("inf")

        if _contains_token(generated, structured_ids["think_end"]):
            token_id = structured_ids["think_end"]
            if token_id is not None:
                next_token_logits[:, token_id] = -float("inf")

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break
        if prompt_type == ACOUSTIC_CUE_PROMPT_TYPE:
            if next_token.item() == structured_ids["caption_end"]:
                break
        elif generate_evidence:
            if next_token.item() == structured_ids["evidence_end"]:
                break
        elif next_token.item() == structured_ids["answer_end"]:
            break

    return generated


def evaluate(config):
    if not os.path.exists(config["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint_path']}")

    if not os.path.exists(config["embeddings_path"]):
        raise FileNotFoundError(
            f"Embeddings file not found: {config['embeddings_path']}. "
            f"Run {config['preprocessing_script']} first."
        )

    device = config["device"]

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )

    checkpoint_config = checkpoint.get("config", {})

    adapter_dim = checkpoint_config.get("adapter_dim", config["adapter_dim"])
    dropout = checkpoint_config.get("dropout", config["dropout"])
    lora_rank = checkpoint.get("lora_rank", checkpoint_config.get("lora_rank", 0))

    dataset = EmoDBGenerationDataset(
        embeddings_path=config["embeddings_path"],
        prompt_type=config["prompt_type"],
        max_length=config["max_prompt_length"],
        speaker_baseline_mode=config["speaker_baseline_mode"],
    )
    label_names = [dataset.idx2label[i] for i in range(len(dataset.idx2label))]

    _, _, test_idx = speaker_independent_split(
        dataset,
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
    )
    prompt_overrides = build_eval_prompt_overrides(
        dataset,
        test_idx,
        config["cue_perturbation"],
    )

    test_dataset = Subset(dataset, test_idx)

    loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    tokenizer = build_generation_tokenizer(verbose=True)

    audio_dim = dataset.embeddings[0].shape[-1]

    model = AudioGPT2Generation(
        audio_dim=audio_dim,
        adapter_dim=adapter_dim,
        dropout=dropout,
        lora_rank=lora_rank,
    ).to(device)
    model.configure_tokenizer_vocab(len(tokenizer))

    load_generation_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    compressor = AudioCompressor(target_len=config["target_audio_len"]).to(device)

    y_true = []
    y_pred = []
    generated_answers = []
    generated_full_texts = []
    sample_rows = []
    format_valid_flags = []
    evidence_format_valid_flags = []
    evidence_non_empty_flags = []
    evidence_cue_mention_flags = []
    evidence_faithfulness_flags = []
    cue_format_valid_flags = []
    cue_correct_flags = {cue_name: [] for cue_name in CUE_NAMES}
    cue_exact_match_flags = []

    print("\nGeneration evaluation configuration:")
    print(f"  Dataset:     {config['dataset']}")
    print(f"  Encoder:     {config['encoder']}")
    print(f"  Prompt type: {config['prompt_type']}")
    print(f"  Device:      {device}")
    print(f"  Checkpoint:  {config['checkpoint_path']}")
    print(f"  Candidate scoring: {config['candidate_scoring']}")
    print(f"  Generate evidence: {config['generate_evidence']}")
    print(f"  No audio:    {config['no_audio']}")
    print(f"  Cue perturbation: {config['cue_perturbation']}")
    print(f"  Speaker baseline mode: {config['speaker_baseline_mode']}")
    print(f"  Class-weighted checkpoint: {config['class_weighted_answer_loss']}")
    print(f"  Max new tokens: {config['max_new_tokens']}")
    print()

    for local_batch_idx, batch in enumerate(loader):
        original_idx = test_idx[local_batch_idx]

        prompt = prompt_overrides.get(
            original_idx,
            build_prompt_for_eval(dataset, original_idx),
        )

        encoded_prompt = tokenizer(
            prompt,
            max_length=config["max_prompt_length"],
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_prompt["input_ids"].to(device)
        class_label = batch["class_label"].item()

        audio_compressed = None
        if not config["no_audio"]:
            audio = batch["audio"].to(device)
            audio_compressed = compressor(audio)

        generated_ids = greedy_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            audio_hidden=audio_compressed,
            max_new_tokens=config["max_new_tokens"],
            prompt_type=config["prompt_type"],
            generate_evidence=config["generate_evidence"],
            label_names=label_names,
        )

        full_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=False,
        )

        if full_text.startswith(prompt):
            generated_text = full_text[len(prompt):].strip()
        else:
            generated_text = full_text.strip()

        true_label = dataset.idx2label[int(class_label)]
        cue_text = extract_speaker_relative_cues(prompt)
        evidence_text, evidence_format_valid = extract_evidence_text(generated_text)
        evidence_non_empty = bool(evidence_text.strip())
        evidence_cue_mention = mentions_evidence_cue(evidence_text)
        evidence_faithfulness = cue_faithfulness_ok(cue_text, evidence_text)

        if config["prompt_type"] == ACOUSTIC_CUE_PROMPT_TYPE:
            true_cues = dataset.build_acoustic_cue_target_for_sample(original_idx)
            pred_cues, cue_format_valid = parse_acoustic_cues(generated_text)
            cue_format_valid_flags.append(cue_format_valid)

            exact_match = True
            for cue_name in CUE_NAMES:
                is_correct = pred_cues[cue_name] == true_cues[cue_name]
                cue_correct_flags[cue_name].append(is_correct)
                exact_match = exact_match and is_correct
            cue_exact_match_flags.append(exact_match)

            generated_answers.append(generated_text)
            generated_full_texts.append(full_text)
            sample_rows.append({
                "prompt_type": config["prompt_type"],
                "true_label": true_label,
                "parsed_prediction": "",
                "speaker_relative_cues": cue_text,
                "generated_text": generated_text,
                "think_text": "",
                "answer_text": "",
                "evidence_text": "",
                "evidence_format_valid": "",
                "evidence_non_empty": "",
                "evidence_cue_mention": "",
                "evidence_faithfulness_ok": "",
                "true_pitch": true_cues["pitch"],
                "pred_pitch": pred_cues["pitch"],
                "true_energy": true_cues["energy"],
                "pred_energy": pred_cues["energy"],
                "true_rhythm": true_cues["rhythm"],
                "pred_rhythm": pred_cues["rhythm"],
                "true_duration": true_cues["duration"],
                "pred_duration": pred_cues["duration"],
                "cue_format_valid": cue_format_valid,
                "cue_exact_match": exact_match,
            })
            continue

        pred_label, think_text, extracted_answer_text, format_valid = parse_generated_label(
            generated_text,
            config["prompt_type"],
            label_names,
        )

        y_true.append(true_label)

        if pred_label is None:
            y_pred.append("invalid")
        else:
            y_pred.append(pred_label)

        generated_answers.append(generated_text)
        generated_full_texts.append(full_text)
        format_valid_flags.append(format_valid)
        evidence_format_valid_flags.append(evidence_format_valid)
        evidence_non_empty_flags.append(evidence_non_empty)
        evidence_cue_mention_flags.append(evidence_cue_mention)
        evidence_faithfulness_flags.append(evidence_faithfulness)
        sample_rows.append({
            "prompt_type": config["prompt_type"],
            "true_label": true_label,
            "parsed_prediction": pred_label or "invalid",
            "speaker_relative_cues": cue_text,
            "generated_text": generated_text,
            "think_text": think_text,
            "answer_text": extracted_answer_text,
            "evidence_text": evidence_text,
            "evidence_format_valid": evidence_format_valid,
            "evidence_non_empty": evidence_non_empty,
            "evidence_cue_mention": evidence_cue_mention,
            "evidence_faithfulness_ok": evidence_faithfulness,
            "true_pitch": "",
            "pred_pitch": "",
            "true_energy": "",
            "pred_energy": "",
            "true_rhythm": "",
            "pred_rhythm": "",
            "true_duration": "",
            "pred_duration": "",
            "cue_format_valid": "",
            "cue_exact_match": "",
        })

    if config["prompt_type"] == ACOUSTIC_CUE_PROMPT_TYPE:
        cue_format_validity = sum(cue_format_valid_flags) / max(
            len(cue_format_valid_flags),
            1,
        )
        cue_accuracies = {
            cue_name: sum(flags) / max(len(flags), 1)
            for cue_name, flags in cue_correct_flags.items()
        }
        macro_cue_accuracy = sum(cue_accuracies.values()) / len(cue_accuracies)
        exact_all_cue_match = sum(cue_exact_match_flags) / max(
            len(cue_exact_match_flags),
            1,
        )

        print("Acoustic cue generation results:")
        print(f"  Format validity:       {cue_format_validity:.4f}")
        print(f"  Pitch accuracy:        {cue_accuracies['pitch']:.4f}")
        print(f"  Energy accuracy:       {cue_accuracies['energy']:.4f}")
        print(f"  Rhythm accuracy:       {cue_accuracies['rhythm']:.4f}")
        print(f"  Duration accuracy:     {cue_accuracies['duration']:.4f}")
        print(f"  Macro cue accuracy:    {macro_cue_accuracy:.4f}")
        print(f"  Exact all-cue match:   {exact_all_cue_match:.4f}")

        print()
        print("Example generations:")
        for i in range(min(10, len(generated_answers))):
            row = sample_rows[i]
            print("-" * 80)
            print(
                "True cues: "
                f"pitch={row['true_pitch']} energy={row['true_energy']} "
                f"rhythm={row['true_rhythm']} duration={row['true_duration']}"
            )
            print(
                "Pred cues: "
                f"pitch={row['pred_pitch']} energy={row['pred_energy']} "
                f"rhythm={row['pred_rhythm']} duration={row['pred_duration']}"
            )
            print(f"Generated: {generated_answers[i]}")

        sample_path = write_generation_samples(config, sample_rows)
        print()
        print(f"Saved generated output samples to: {sample_path}")
        return

    invalid_count = sum(1 for pred in y_pred if pred == "invalid")
    format_validity = sum(format_valid_flags) / max(len(format_valid_flags), 1)

    print("Free-generation results:")
    _print_classification_metrics(y_true, y_pred, label_names)
    print(f"  Format validity:          {format_validity:.4f}")

    if config["generate_evidence"]:
        evidence_format_validity = sum(evidence_format_valid_flags) / max(
            len(evidence_format_valid_flags),
            1,
        )
        evidence_non_empty_rate = sum(evidence_non_empty_flags) / max(
            len(evidence_non_empty_flags),
            1,
        )
        evidence_cue_mention_rate = sum(evidence_cue_mention_flags) / max(
            len(evidence_cue_mention_flags),
            1,
        )
        evidence_faithfulness_rate = sum(evidence_faithfulness_flags) / max(
            len(evidence_faithfulness_flags),
            1,
        )

        print()
        print("Evidence diagnostics:")
        print(f"  Evidence format validity: {evidence_format_validity:.4f}")
        print(f"  Evidence non-empty rate:  {evidence_non_empty_rate:.4f}")
        print(f"  Cue mention rate:         {evidence_cue_mention_rate:.4f}")
        print(f"  Cue faithfulness rate:    {evidence_faithfulness_rate:.4f}")

    print()
    print("Example generations:")
    example_count = 20 if config["generate_evidence"] else 10
    for i in range(min(example_count, len(generated_answers))):
        print("-" * 80)
        print(f"True:      {y_true[i]}")
        print(f"Pred:      {y_pred[i]}")
        if config["generate_evidence"]:
            print(f"Cues:      {sample_rows[i]['speaker_relative_cues']}")
            print(f"Evidence:  {sample_rows[i]['evidence_text']}")
        print(f"Generated: {generated_answers[i]}")

    sample_path = write_generation_samples(config, sample_rows)

    print()
    print(f"Saved generated output samples to: {sample_path}")

    if config["candidate_scoring"] in ("answer", "both"):
        evaluate_candidate_scoring(
            model=model,
            tokenizer=tokenizer,
            compressor=compressor,
            dataset=dataset,
            test_idx=test_idx,
            loader=loader,
            config=config,
            mode="answer",
            label_names=label_names,
            prompt_overrides=prompt_overrides,
        )

    if config["candidate_scoring"] in ("full_target", "both"):
        evaluate_candidate_scoring(
            model=model,
            tokenizer=tokenizer,
            compressor=compressor,
            dataset=dataset,
            test_idx=test_idx,
            loader=loader,
            config=config,
            mode="full_target",
            label_names=label_names,
            prompt_overrides=prompt_overrides,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="emodb",
        choices=list(DATASET_CONFIGS),
        help="Which dataset's embeddings, labels, and speaker split to use.",
    )

    parser.add_argument(
        "--encoder",
        default="wavlm-large",
        choices=[
            "wav2vec2-base",
            "wav2vec2-large-emotion",
            "wavlm-large",
            "hubert-large",
        ],
        help="Which encoder's embeddings to evaluate on.",
    )

    parser.add_argument(
        "--prompt_type",
        default="generation",
        choices=GENERATION_PROMPT_TYPES,
        help="Generation prompt template to evaluate.",
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional path to generation checkpoint. If not set, uses the default checkpoint name.",
    )

    parser.add_argument(
        "--candidate_scoring",
        default="none",
        choices=["none", "answer", "full_target", "both"],
        help="Optional candidate label likelihood scoring mode.",
    )

    parser.add_argument(
        "--generate_evidence",
        action="store_true",
        help=(
            "For speaker_feature_answer_evidence_generation, continue decoding "
            "after </answer> and evaluate generated evidence text."
        ),
    )

    parser.add_argument(
        "--no_audio",
        action="store_true",
        help="Evaluate generation with text prompts only and no audio fusion.",
    )

    parser.add_argument(
        "--cue_perturbation",
        default="none",
        choices=["none", "invert", "shuffle"],
        help="Evaluation-only perturbation for speaker-relative cue text prompts.",
    )

    parser.add_argument(
        "--speaker_baseline_mode",
        choices=["neutral", "emotion_balanced"],
        default="neutral",
        help=(
            "Speaker-relative baseline enrollment mode. Use emotion_balanced "
            "to reproduce the old one-utterance-per-emotion enrollment behavior."
        ),
    )

    parser.add_argument(
        "--class_weighted_answer_loss",
        action="store_true",
        help=(
            "Use the class-weighted answer-loss checkpoint naming variant. "
            "This affects default checkpoint selection only during evaluation."
        ),
    )

    parser.add_argument(
        "--class_weight_mode",
        choices=["balanced", "inverse"],
        default="inverse",
        help=(
            "Class weighting mode used by the checkpoint naming variant. "
            "'inverse' is Andreas's training-split inverse-frequency formula "
            "weight_c = total_count / count_c; 'balanced' remains available "
            "as an optional alternative."
        ),
    )

    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=1.0,
        help="Class weight power used by the checkpoint naming variant.",
    )

    parser.add_argument(
        "--class_weight_max",
        type=float,
        default=5.0,
        help="Class weight max used by the checkpoint naming variant.",
    )

    args = parser.parse_args()

    config = _build_config(
        encoder=args.encoder,
        dataset=args.dataset,
        prompt_type=args.prompt_type,
        checkpoint_path=args.checkpoint_path,
        candidate_scoring=args.candidate_scoring,
        generate_evidence=args.generate_evidence,
        no_audio=args.no_audio,
        cue_perturbation=args.cue_perturbation,
        speaker_baseline_mode=args.speaker_baseline_mode,
        class_weighted_answer_loss=args.class_weighted_answer_loss,
        class_weight_mode=args.class_weight_mode,
        class_weight_power=args.class_weight_power,
        class_weight_max=args.class_weight_max,
    )

    evaluate(config)
