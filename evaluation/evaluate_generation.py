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

from data.generation_dataset import EmoDBGenerationDataset
from data.dataset import speaker_independent_split
from data.prompts import LABELS
from data.tokenizer_utils import build_generation_tokenizer
from models.compression.compressor import AudioCompressor
from models.audio_gpt2_generation import AudioGPT2Generation


GENERATION_PROMPT_TYPES = [
    "generation",
    "feature_generation",
    "answer_generation",
    "speaker_feature_answer_generation",
    "reasoning_generation_global",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
]

REASONING_PROMPT_TYPES = {
    "reasoning_generation_global",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
}

ANSWER_TAG_PROMPT_TYPES = REASONING_PROMPT_TYPES | {
    "answer_generation",
    "speaker_feature_answer_generation",
}

LABEL_CONSTRAINED_PROMPT_TYPES = {
    "answer_generation",
    "speaker_feature_answer_generation",
}


def _max_length_for_prompt_type(prompt_type: str) -> int:
    if "reasoning_generation" in prompt_type:
        return 224
    if "feature" in prompt_type:
        return 128
    return 96


def _max_new_tokens_for_prompt_type(prompt_type: str) -> int:
    if prompt_type in REASONING_PROMPT_TYPES:
        return 96
    return 5


def _build_config(
    encoder: str,
    prompt_type: str,
    checkpoint_path: str | None = None,
    candidate_scoring: str = "none",
) -> dict:
    tag = f"{encoder}_{prompt_type}_generation"

    return {
        "encoder": encoder,
        "prompt_type": prompt_type,
        "max_prompt_length": _max_length_for_prompt_type(prompt_type),
        "embeddings_path": f"embeddings/{encoder}_embeddings.pt",
        "batch_size": 1,
        "adapter_dim": 64,
        "dropout": 0.3,
        "target_audio_len": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_speakers": ["09", "10"],
        "test_speakers": ["03", "08"],
        "checkpoint_path": checkpoint_path or f"checkpoints/{tag}_best.pt",
        "max_new_tokens": _max_new_tokens_for_prompt_type(prompt_type),
        "candidate_scoring": candidate_scoring,
    }


def normalize_generated_label(text: str) -> str | None:
    """
    Map generated text back to one of the emotion labels.

    Example:
        " anger" -> "anger"
        "The emotion is anger." -> "anger"
        "happy" -> None, because the label is "happiness"
    """
    text = text.lower().strip()

    for label in LABELS:
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


def parse_generated_label(text: str, prompt_type: str) -> tuple[str | None, str, str, bool]:
    think_text, answer_text, format_valid = extract_reasoning_blocks(text)

    if prompt_type in ANSWER_TAG_PROMPT_TYPES and answer_text:
        pred_label = normalize_generated_label(answer_text)
    else:
        pred_label = normalize_generated_label(text)

    if pred_label is None and answer_text:
        pred_label = normalize_generated_label(answer_text)

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
):
    y_true = []
    y_pred = []

    for local_batch_idx, batch in enumerate(loader):
        original_idx = test_idx[local_batch_idx]
        prompt = build_prompt_for_eval(dataset, original_idx)
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

        for label in LABELS:
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
    _print_classification_metrics(y_true, y_pred)


def _print_classification_metrics(y_true, y_pred):
    all_eval_labels = LABELS + ["invalid"]

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
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    print("Per-class F1:")
    for label, score in zip(LABELS, per_class_f1):
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


def _single_token_id(tokenizer, token: str) -> int | None:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def _structured_token_ids(tokenizer):
    return {
        "think_start": _single_token_id(tokenizer, "<think>"),
        "think_end": _single_token_id(tokenizer, "</think>"),
        "answer_start": _single_token_id(tokenizer, "<answer>"),
        "answer_end": _single_token_id(tokenizer, "</answer>"),
    }


def _contains_token(input_ids, token_id: int | None) -> bool:
    if token_id is None:
        return False
    return bool(input_ids.eq(token_id).any().item())


def _label_token_sequences(tokenizer):
    return {
        label: tokenizer.encode(label, add_special_tokens=False)
        for label in LABELS
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
    label_sequences = _label_token_sequences(tokenizer)

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

        if _contains_token(generated, structured_ids["answer_start"]):
            token_id = structured_ids["answer_start"]
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
        if next_token.item() == structured_ids["answer_end"]:
            break

    return generated


def evaluate(config):
    if not os.path.exists(config["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint_path']}")

    if not os.path.exists(config["embeddings_path"]):
        raise FileNotFoundError(f"Embeddings file not found: {config['embeddings_path']}")

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
    )

    _, _, test_idx = speaker_independent_split(
        dataset,
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
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

    print("\nGeneration evaluation configuration:")
    print(f"  Encoder:     {config['encoder']}")
    print(f"  Prompt type: {config['prompt_type']}")
    print(f"  Device:      {device}")
    print(f"  Checkpoint:  {config['checkpoint_path']}")
    print(f"  Candidate scoring: {config['candidate_scoring']}")
    print()

    for local_batch_idx, batch in enumerate(loader):
        original_idx = test_idx[local_batch_idx]

        prompt = build_prompt_for_eval(dataset, original_idx)

        encoded_prompt = tokenizer(
            prompt,
            max_length=config["max_prompt_length"],
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_prompt["input_ids"].to(device)
        audio = batch["audio"].to(device)
        class_label = batch["class_label"].item()

        audio_compressed = compressor(audio)

        generated_ids = greedy_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            audio_hidden=audio_compressed,
            max_new_tokens=config["max_new_tokens"],
            prompt_type=config["prompt_type"],
        )

        full_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=False,
        )

        if full_text.startswith(prompt):
            generated_text = full_text[len(prompt):].strip()
        else:
            generated_text = full_text.strip()

        pred_label, think_text, extracted_answer_text, format_valid = parse_generated_label(
            generated_text,
            config["prompt_type"],
        )

        true_label = dataset.idx2label[int(class_label)]

        y_true.append(true_label)

        if pred_label is None:
            y_pred.append("invalid")
        else:
            y_pred.append(pred_label)

        generated_answers.append(generated_text)
        generated_full_texts.append(full_text)
        format_valid_flags.append(format_valid)
        sample_rows.append({
            "prompt_type": config["prompt_type"],
            "true_label": true_label,
            "parsed_prediction": pred_label or "invalid",
            "generated_text": generated_text,
            "think_text": think_text,
            "answer_text": extracted_answer_text,
        })

    invalid_count = sum(1 for pred in y_pred if pred == "invalid")
    format_validity = sum(format_valid_flags) / max(len(format_valid_flags), 1)

    print("Free-generation results:")
    _print_classification_metrics(y_true, y_pred)
    print(f"  Format validity:          {format_validity:.4f}")

    print()
    print("Example generations:")
    for i in range(min(10, len(generated_answers))):
        print("-" * 80)
        print(f"True:      {y_true[i]}")
        print(f"Pred:      {y_pred[i]}")
        print(f"Generated: {generated_answers[i]}")

    os.makedirs("evaluation_outputs", exist_ok=True)
    sample_path = os.path.join(
        "evaluation_outputs",
        f"{config['encoder']}_{config['prompt_type']}_samples.csv",
    )
    with open(sample_path, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "prompt_type",
                "true_label",
                "parsed_prediction",
                "generated_text",
                "think_text",
                "answer_text",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows[:50])

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
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    config = _build_config(
        encoder=args.encoder,
        prompt_type=args.prompt_type,
        checkpoint_path=args.checkpoint_path,
        candidate_scoring=args.candidate_scoring,
    )

    evaluate(config)
