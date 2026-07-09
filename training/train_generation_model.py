import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from data.dataset_configs import DATASET_CONFIGS, get_dataset_config
from data.generation_dataset import (
    BASELINE_ESTIMATION_MODES,
    EmoDBGenerationDataset,
    SPEAKER_BASELINE_PROMPT_TYPES,
    print_baseline_estimation_summary,
)
from data.dataset import speaker_independent_split
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

STRUCTURED_ANSWER_PROMPT_TYPES = {
    "answer_generation",
    "speaker_feature_answer_generation",
    "speaker_feature_answer_caption_generation",
    "speaker_feature_answer_evidence_generation",
    "reasoning_generation_global",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
}


def _checkpoint_tag(
    encoder: str,
    prompt_type: str,
    speaker_baseline_mode: str,
    baseline_estimation_mode: str = "speaker_neutral",
    class_weighted_answer_loss: bool = False,
    class_weight_mode: str = "inverse",
    class_weight_power: float = 0.5,
    class_weight_max: float = 2.0,
) -> str:
    tag = f"{encoder}_{prompt_type}"

    if prompt_type in SPEAKER_BASELINE_PROMPT_TYPES:
        tag += f"_{speaker_baseline_mode}"

    if baseline_estimation_mode == "mixed_effects":
        tag += "_mixed_effects"

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
    if prompt_type == "speaker_acoustic_cue_generation":
        return 128
    if "reasoning_generation" in prompt_type:
        return 224
    if "feature" in prompt_type:
        return 128
    return 96


def _build_config(
    encoder: str,
    dataset: str = "emodb",
    prompt_type: str = "generation",
    lora_rank: int = 0,
    lora_lr: float = 1e-4,
    epochs: int = 100,
    answer_loss_weight: float = 5.0,
    evidence_loss_weight: float = 0.3,
    class_weighted_answer_loss: bool = False,
    class_weight_mode: str = "inverse",
    class_weight_power: float = 0.5,
    class_weight_max: float = 2.0,
    no_audio: bool = False,
    speaker_baseline_mode: str = "neutral",
    baseline_estimation_mode: str = "speaker_neutral",
) -> dict:
    dataset_config = get_dataset_config(dataset)
    tag = _checkpoint_tag(
        encoder=encoder,
        prompt_type=prompt_type,
        speaker_baseline_mode=speaker_baseline_mode,
        baseline_estimation_mode=baseline_estimation_mode,
        class_weighted_answer_loss=class_weighted_answer_loss,
        class_weight_mode=class_weight_mode,
        class_weight_power=class_weight_power,
        class_weight_max=class_weight_max,
    )

    if lora_rank > 0:
        tag += f"_lora{lora_rank}"

    checkpoint_dir = dataset_config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    return {
        "dataset": dataset,
        "encoder": encoder,
        "prompt_type": prompt_type,
        "max_prompt_length": _max_length_for_prompt_type(prompt_type),
        "lora_rank": lora_rank,
        "lora_lr": lora_lr,
        "answer_loss_weight": answer_loss_weight,
        "evidence_loss_weight": evidence_loss_weight,
        "class_weighted_answer_loss": class_weighted_answer_loss,
        "class_weight_mode": class_weight_mode,
        "class_weight_power": class_weight_power,
        "class_weight_max": class_weight_max,
        "no_audio": no_audio,
        "speaker_baseline_mode": speaker_baseline_mode,
        "baseline_estimation_mode": baseline_estimation_mode,
        "embeddings_path": (
            f"embeddings/{dataset_config['embeddings_prefix']}"
            f"{encoder}_embeddings.pt"
        ),
        "batch_size": 4,
        "lr": 1e-5,
        "epochs": epochs,
        "adapter_dim": 64,
        "dropout": 0.3,
        "target_audio_len": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_speakers": dataset_config["val_speakers"],
        "test_speakers": dataset_config["test_speakers"],
        "checkpoint_path": f"{checkpoint_dir}/{tag}_best.pt",
        "preprocessing_script": dataset_config["preprocessing_script"],
    }


def smoke_test(config):
    audio_dim = 768
    prompt_len = config["max_prompt_length"]
    tokenizer = build_generation_tokenizer(verbose=True)

    input_ids = torch.randint(0, len(tokenizer), (2, prompt_len))
    audio = torch.randn(2, 50, audio_dim)

    model = AudioGPT2Generation(
        audio_dim=audio_dim,
        adapter_dim=config["adapter_dim"],
        dropout=config["dropout"],
        lora_rank=config["lora_rank"],
    )
    model.configure_tokenizer_vocab(len(tokenizer))

    logits = model(input_ids, None if config["no_audio"] else audio)

    assert logits.shape[0] == 2, f"Expected batch size 2, got {logits.shape[0]}"
    assert logits.shape[1] == prompt_len, f"Expected seq len {prompt_len}, got {logits.shape[1]}"
    assert logits.shape[2] == len(tokenizer), (
        f"Expected tokenizer vocab size {len(tokenizer)}, got {logits.shape[2]}"
    )

    print("✓ Generation smoke test passed")


def train(config):
    if not os.path.exists(config["embeddings_path"]):
        print(
            f"ERROR: '{config['embeddings_path']}' not found. "
            f"Run {config['preprocessing_script']} first to generate the embeddings file."
        )
        sys.exit(1)

    dataset = EmoDBGenerationDataset(
        embeddings_path=config["embeddings_path"],
        prompt_type=config["prompt_type"],
        max_length=config["max_prompt_length"],
        answer_loss_weight=config["answer_loss_weight"],
        evidence_loss_weight=config["evidence_loss_weight"],
        speaker_baseline_mode=config["speaker_baseline_mode"],
    )

    if (
        config["class_weighted_answer_loss"]
        and config["prompt_type"] not in STRUCTURED_ANSWER_PROMPT_TYPES
    ):
        raise ValueError(
            "--class_weighted_answer_loss requires a prompt type with "
            "structured <answer>...</answer> targets."
        )

    train_idx, val_idx, test_idx = speaker_independent_split(
        dataset,
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
    )

    baseline_summary = dataset.apply_baseline_estimation(
        config["baseline_estimation_mode"],
        train_idx,
    )
    config["baseline_estimation_summary"] = baseline_summary
    print_baseline_estimation_summary(baseline_summary)
    print(f"Checkpoint path: {config['checkpoint_path']}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    device = config["device"]

    compressor = AudioCompressor(target_len=config["target_audio_len"]).to(device)

    audio_dim = dataset.embeddings[0].shape[-1]

    model = AudioGPT2Generation(
        audio_dim=audio_dim,
        adapter_dim=config["adapter_dim"],
        dropout=config["dropout"],
        lora_rank=config["lora_rank"],
    ).to(device)
    model.configure_tokenizer_vocab(len(dataset.tokenizer))

    answer_class_weights = None
    if config["class_weighted_answer_loss"]:
        answer_class_weights = compute_class_weights(
            dataset=dataset,
            train_idx=train_idx,
            mode=config["class_weight_mode"],
            power=config["class_weight_power"],
            max_weight=config["class_weight_max"],
            device=device,
        )

    use_loss_weights = (
        config["prompt_type"] == "speaker_feature_answer_evidence_generation"
        or answer_class_weights is not None
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="none" if use_loss_weights else "mean",
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if config["lora_rank"] > 0:
        lora_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and n.endswith((".A", ".B"))
        ]

        other_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and not n.endswith((".A", ".B"))
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": config["lr"]},
                {"params": lora_params, "lr": config["lora_lr"]},
            ],
            weight_decay=1e-2,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["lr"],
            weight_decay=1e-2,
        )

    epochs = config["epochs"]
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    print("\nGeneration training configuration:")
    print(f"  Dataset:      {config['dataset']}")
    print(f"  Encoder:      {config['encoder']}")
    print(f"  Prompt type:  {config['prompt_type']}")
    print(f"  Prompt length:{config['max_prompt_length']}")
    print(f"  Speaker baseline mode: {config['speaker_baseline_mode']}")
    print(f"  Baseline estimation mode: {config['baseline_estimation_mode']}")
    print(f"  LoRA rank:    {config['lora_rank']}")
    if config["prompt_type"] == "speaker_feature_answer_evidence_generation":
        print(f"  Answer weight:{config['answer_loss_weight']}")
        print(f"  Evidence wt:  {config['evidence_loss_weight']}")
    if answer_class_weights is not None:
        print("  Class-weighted answer loss: enabled")
        print(f"  Class weight mode:  {config['class_weight_mode']}")
        print(f"  Class weight power: {config['class_weight_power']}")
        print(f"  Class weight max:   {config['class_weight_max']}")
        print("  Answer class weights:")
        for idx, weight in enumerate(answer_class_weights.detach().cpu().tolist()):
            print(f"    {dataset.idx2label[idx]}: {weight:.4f}")
    print(f"  No audio:     {config['no_audio']}")
    print(f"  Device:       {device}")
    print(f"  Checkpoint:   {config['checkpoint_path']}")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_weights = build_batch_loss_weights(
                batch=batch,
                labels=labels,
                device=device,
                answer_class_weights=answer_class_weights,
            )
            audio_compressed = None
            if not config["no_audio"]:
                audio = batch["audio"].to(device)
                audio_compressed = compressor(audio)

            logits = model(input_ids, audio_compressed)

            loss = language_model_loss(logits, labels, criterion, loss_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                loss_weights = build_batch_loss_weights(
                    batch=batch,
                    labels=labels,
                    device=device,
                    answer_class_weights=answer_class_weights,
                )
                audio_compressed = None
                if not config["no_audio"]:
                    audio = batch["audio"].to(device)
                    audio_compressed = compressor(audio)

                logits = model(input_ids, audio_compressed)

                loss = language_model_loss(logits, labels, criterion, loss_weights)

                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train_loss: {epoch_train_loss:.4f} | "
            f"val_loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "dataset": config["dataset"],
                    "encoder": config["encoder"],
                    "prompt_type": config["prompt_type"],
                    "max_prompt_length": config["max_prompt_length"],
                    "lora_rank": config["lora_rank"],
                    "class_weighted_answer_loss": (
                        config["class_weighted_answer_loss"]
                    ),
                    "class_weight_mode": config["class_weight_mode"],
                    "class_weight_power": config["class_weight_power"],
                    "class_weight_max": config["class_weight_max"],
                    "no_audio": config["no_audio"],
                    "speaker_baseline_mode": config["speaker_baseline_mode"],
                    "baseline_estimation_mode": config["baseline_estimation_mode"],
                    "baseline_estimation_summary": baseline_summary,
                    "answer_class_weights": (
                        answer_class_weights.detach().cpu()
                        if answer_class_weights is not None
                        else None
                    ),
                    "model_state_dict": model.state_dict(),
                    "val_loss": epoch_val_loss,
                    "idx2label": dataset.idx2label,
                    "label2idx": dataset.label2idx,
                    "config": config,
                },
                config["checkpoint_path"],
            )

            print(f"  ✓ Saved best generation checkpoint (val_loss: {epoch_val_loss:.4f})")

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )

    print(f"\nBest checkpoint saved at epoch {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint['val_loss']:.4f}")

    test_loss = evaluate_loss(
        model=model,
        compressor=compressor,
        loader=test_loader,
        criterion=criterion,
        device=device,
        answer_class_weights=answer_class_weights,
        no_audio=config["no_audio"],
    )

    print(f"Test LM loss: {test_loss:.4f}")


def evaluate_loss(
    model,
    compressor,
    loader,
    criterion,
    device,
    answer_class_weights=None,
    no_audio=False,
):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_weights = build_batch_loss_weights(
                batch=batch,
                labels=labels,
                device=device,
                answer_class_weights=answer_class_weights,
            )
            audio_compressed = None
            if not no_audio:
                audio = batch["audio"].to(device)
                audio_compressed = compressor(audio)

            logits = model(input_ids, audio_compressed)

            loss = language_model_loss(logits, labels, criterion, loss_weights)

            total_loss += loss.item()

    return total_loss / len(loader)


def compute_class_weights(
    dataset,
    train_idx,
    mode: str,
    power: float,
    max_weight: float,
    device,
):
    """Compute answer-label class weights from the training split only.

    The default ``inverse`` mode follows Andreas's requested formula:
    weight_c = total_count / count_c. ``balanced`` remains available as the
    conventional sklearn-style alternative.
    """
    class_labels = torch.tensor(
        [dataset.class_labels_list[idx].item() for idx in train_idx],
        dtype=torch.long,
    )
    num_classes = len(dataset.idx2label)
    counts = torch.bincount(class_labels, minlength=num_classes).float()
    total = counts.sum().clamp_min(1.0)

    weights = torch.ones(num_classes, dtype=torch.float)
    present = counts > 0
    if mode == "balanced":
        weights[present] = total / (float(num_classes) * counts[present])
    elif mode == "inverse":
        weights[present] = total / counts[present]
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")

    if power != 1.0:
        weights = weights.pow(power)

    if max_weight is not None and max_weight > 0:
        weights = weights.clamp(max=max_weight)

    return weights.to(device)


def build_batch_loss_weights(batch, labels, device, answer_class_weights=None):
    loss_weights = batch.get("loss_weights")

    if loss_weights is not None:
        loss_weights = loss_weights.to(device)
    elif answer_class_weights is not None:
        loss_weights = torch.ones_like(labels, dtype=torch.float, device=device)

    if answer_class_weights is None:
        return loss_weights

    answer_loss_mask = batch.get("answer_loss_mask")
    if answer_loss_mask is None:
        return loss_weights

    if loss_weights is None:
        loss_weights = torch.ones_like(labels, dtype=torch.float, device=device)

    answer_loss_mask = answer_loss_mask.to(device)
    class_labels = batch["class_label"].to(device)
    sample_weights = answer_class_weights[class_labels].view(-1, 1)

    return loss_weights * (1.0 - answer_loss_mask) + (
        loss_weights * sample_weights * answer_loss_mask
    )


def language_model_loss(logits, labels, criterion, loss_weights=None):
    """
    Decoder-only LM loss: token t predicts token t+1.

    The dataset masks prompt positions with -100. After shifting, the first
    target token is predicted from the final prompt position, which is the
    behavior needed for generation.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    token_loss = criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    if loss_weights is None:
        return token_loss

    shift_weights = loss_weights[:, 1:].contiguous().view(-1)
    active = shift_labels.view(-1) != -100
    active_weights = shift_weights * active.float()

    return (token_loss * active_weights).sum() / active_weights.sum().clamp_min(1e-8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="emodb",
        choices=list(DATASET_CONFIGS),
        help="Which dataset's embeddings and speaker split to use.",
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
        help="Which encoder's embeddings to train on.",
    )

    parser.add_argument(
        "--prompt_type",
        default="generation",
        choices=GENERATION_PROMPT_TYPES,
        help="Generation prompt template to use.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="LoRA rank for GPT-2 attention layers (0 = disabled).",
    )

    parser.add_argument(
        "--lora_lr",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA parameters.",
    )

    parser.add_argument(
        "--answer_loss_weight",
        type=float,
        default=5.0,
        help=(
            "Loss weight for the answer span in "
            "speaker_feature_answer_evidence_generation."
        ),
    )

    parser.add_argument(
        "--evidence_loss_weight",
        type=float,
        default=0.3,
        help=(
            "Loss weight for the evidence span in "
            "speaker_feature_answer_evidence_generation."
        ),
    )

    parser.add_argument(
        "--class_weighted_answer_loss",
        action="store_true",
        help=(
            "Apply train-split class weights to answer-label tokens "
            "for structured <answer>...</answer> generation targets."
        ),
    )

    parser.add_argument(
        "--class_weight_mode",
        choices=["balanced", "inverse"],
        default="inverse",
        help=(
            "Class weighting formula for answer-label tokens. 'inverse' uses "
            "Andreas's inverse-frequency formula computed on the training "
            "split, weight_c = total_count / count_c; 'balanced' uses "
            "weight_c = total_count / (num_classes * count_c)."
        ),
    )

    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=0.5,
        help="Exponent applied to answer class weights.",
    )

    parser.add_argument(
        "--class_weight_max",
        type=float,
        default=2.0,
        help="Maximum clipped answer class weight; set <= 0 to disable clipping.",
    )

    parser.add_argument(
        "--no_audio",
        action="store_true",
        help="Train the generation model with text prompts only and no audio fusion.",
    )

    parser.add_argument(
        "--speaker_baseline_mode",
        choices=["neutral", "emotion_balanced"],
        default="neutral",
        help=(
            "Speaker-relative baseline enrollment mode. 'neutral' uses neutral "
            "utterances where available; 'emotion_balanced' restores the old "
            "one-utterance-per-emotion enrollment behavior."
        ),
    )

    parser.add_argument(
        "--baseline_estimation_mode",
        choices=BASELINE_ESTIMATION_MODES,
        default="speaker_neutral",
        help=(
            "Statistical speaker baseline estimator for generation prompts. "
            "'speaker_neutral' preserves the Q1/Q2 fixed enrollment baseline; "
            "'mixed_effects' applies mixed-effects speaker baseline estimation "
            "inside the full audio-conditioned generation model before "
            "generation targets are built."
        ),
    )

    args = parser.parse_args()

    config = _build_config(
        encoder=args.encoder,
        dataset=args.dataset,
        prompt_type=args.prompt_type,
        lora_rank=args.lora_rank,
        lora_lr=args.lora_lr,
        epochs=args.epochs,
        answer_loss_weight=args.answer_loss_weight,
        evidence_loss_weight=args.evidence_loss_weight,
        class_weighted_answer_loss=args.class_weighted_answer_loss,
        class_weight_mode=args.class_weight_mode,
        class_weight_power=args.class_weight_power,
        class_weight_max=args.class_weight_max,
        no_audio=args.no_audio,
        speaker_baseline_mode=args.speaker_baseline_mode,
        baseline_estimation_mode=args.baseline_estimation_mode,
    )

    smoke_test(config)
    train(config)
