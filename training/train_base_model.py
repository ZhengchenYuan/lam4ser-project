import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from data.dataset import EmoDBFusionDataset, speaker_independent_split
from models.compression.compressor import AudioCompressor
from models.audio_gpt2 import AudioGPT2

CONFIG = {
    "embeddings_path": "embeddings/emodb_embeddings.pt",
    "batch_size": 8,
    "lr": 1e-5,
    "epochs": 100,
    "adapter_dim": 64,
    "dropout": 0.3,
    "target_audio_len": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "val_speakers": ["09", "10"],
    "test_speakers": ["03", "08"],
    "checkpoint_path": "best_model.pt",
    "loss_curve_path": "loss_curve.png",
}


def smoke_test():
    input_ids = torch.randint(0, 50256, (2, 32))
    audio = torch.randn(2, 50, 768)

    model = AudioGPT2(num_classes=7, adapter_dim=CONFIG["adapter_dim"], dropout=CONFIG["dropout"])
    logits = model(input_ids, audio)

    assert logits.shape == (2, 7), f"Expected logits shape (2, 7), got {logits.shape}"

    print("✓ Smoke test passed")


def train():
    if not os.path.exists(CONFIG["embeddings_path"]):
        print(
            f"ERROR: '{CONFIG['embeddings_path']}' not found. "
            "Run models/fusion/preprocessing.py first to generate the embeddings file."
        )
        sys.exit(1)

    dataset = EmoDBFusionDataset(CONFIG["embeddings_path"])
    train_idx, val_idx, test_idx = speaker_independent_split(
        dataset,
        val_speakers=CONFIG["val_speakers"],
        test_speakers=CONFIG["test_speakers"],
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=CONFIG["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=CONFIG["batch_size"], shuffle=False
    )

    device = CONFIG["device"]
    compressor = AudioCompressor(target_len=CONFIG["target_audio_len"]).to(device)
    num_classes = len(dataset.label2idx)
    model = AudioGPT2(
        num_classes=num_classes,
        adapter_dim=CONFIG["adapter_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)

    train_labels = [dataset[i]["label"].item() for i in train_idx]
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=train_labels
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=CONFIG["lr"], weight_decay=1e-2
    )

    epochs = CONFIG["epochs"]
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            audio_compressed = compressor(audio)
            logits = model(input_ids, audio_compressed)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)

                audio_compressed = compressor(audio)
                logits = model(input_ids, audio_compressed)
                loss = criterion(logits, labels)

                epoch_val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        epoch_val_loss /= len(val_loader)
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train_loss: {epoch_train_loss:.4f} | "
            f"val_loss: {epoch_val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"val_f1: {val_f1:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": epoch_val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "idx2label": dataset.idx2label,
                },
                CONFIG["checkpoint_path"],
            )
            print(f"  ✓ Saved best checkpoint (val_loss: {epoch_val_loss:.4f})")

    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            audio_compressed = compressor(audio)
            logits = model(input_ids, audio_compressed)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    label_names = [dataset.idx2label[i] for i in range(len(dataset.idx2label))]

    print(f"\nTest results (best checkpoint, epoch {checkpoint['epoch']}):")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  Weighted F1: {test_f1:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {label_names}")
    print(cm)

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(CONFIG["loss_curve_path"])
    plt.close()
    print(f"\nLoss curve saved to {CONFIG['loss_curve_path']}")


if __name__ == "__main__":
    smoke_test()
    train()
