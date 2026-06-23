import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from data.generation_dataset import EmoDBGenerationDataset
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
    "reasoning_generation_global",
    "speaker_reasoning_generation",
    "speaker_reasoning_generation_answer_first",
]


def _max_length_for_prompt_type(prompt_type: str) -> int:
    if "reasoning_generation" in prompt_type:
        return 224
    if "feature" in prompt_type:
        return 128
    return 96


def _build_config(
    encoder: str,
    prompt_type: str = "generation",
    lora_rank: int = 0,
    lora_lr: float = 1e-4,
    epochs: int = 100,
) -> dict:
    tag = f"{encoder}_{prompt_type}_generation"

    if lora_rank > 0:
        tag += f"_lora{lora_rank}"

    os.makedirs("checkpoints", exist_ok=True)

    return {
        "encoder": encoder,
        "prompt_type": prompt_type,
        "max_prompt_length": _max_length_for_prompt_type(prompt_type),
        "lora_rank": lora_rank,
        "lora_lr": lora_lr,
        "embeddings_path": f"embeddings/{encoder}_embeddings.pt",
        "batch_size": 4,
        "lr": 1e-5,
        "epochs": epochs,
        "adapter_dim": 64,
        "dropout": 0.3,
        "target_audio_len": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_speakers": ["09", "10"],
        "test_speakers": ["03", "08"],
        "checkpoint_path": f"checkpoints/{tag}_best.pt",
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

    logits = model(input_ids, audio)

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
            "Run models/audio_encoder/preprocessing.py first to generate the embeddings file."
        )
        sys.exit(1)

    dataset = EmoDBGenerationDataset(
        embeddings_path=config["embeddings_path"],
        prompt_type=config["prompt_type"],
        max_length=config["max_prompt_length"],
    )

    train_idx, val_idx, test_idx = speaker_independent_split(
        dataset,
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
    )

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

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

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
    print(f"  Encoder:      {config['encoder']}")
    print(f"  Prompt type:  {config['prompt_type']}")
    print(f"  Prompt length:{config['max_prompt_length']}")
    print(f"  LoRA rank:    {config['lora_rank']}")
    print(f"  Device:       {device}")
    print(f"  Checkpoint:   {config['checkpoint_path']}")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            audio = batch["audio"].to(device)

            audio_compressed = compressor(audio)

            logits = model(input_ids, audio_compressed)

            loss = language_model_loss(logits, labels, criterion)

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
                audio = batch["audio"].to(device)

                audio_compressed = compressor(audio)

                logits = model(input_ids, audio_compressed)

                loss = language_model_loss(logits, labels, criterion)

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
                    "encoder": config["encoder"],
                    "prompt_type": config["prompt_type"],
                    "max_prompt_length": config["max_prompt_length"],
                    "lora_rank": config["lora_rank"],
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
    )

    print(f"Test LM loss: {test_loss:.4f}")


def evaluate_loss(model, compressor, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            audio = batch["audio"].to(device)

            audio_compressed = compressor(audio)

            logits = model(input_ids, audio_compressed)

            loss = language_model_loss(logits, labels, criterion)

            total_loss += loss.item()

    return total_loss / len(loader)


def language_model_loss(logits, labels, criterion):
    """
    Decoder-only LM loss: token t predicts token t+1.

    The dataset masks prompt positions with -100. After shifting, the first
    target token is predicted from the final prompt position, which is the
    behavior needed for generation.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
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

    args = parser.parse_args()

    config = _build_config(
        encoder=args.encoder,
        prompt_type=args.prompt_type,
        lora_rank=args.lora_rank,
        lora_lr=args.lora_lr,
        epochs=args.epochs,
    )

    smoke_test(config)
    train(config)
