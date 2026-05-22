"""
Dry-run of the full training loop on synthetic data.
No embeddings file needed. Run from the project root:
    python tests/test_training_loop.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from models.compression.compressor import AudioCompressor
from models.audio_gpt2 import AudioGPT2

NUM_CLASSES = 7
T_AUDIO = 399
EPOCHS = 2
BATCH_SIZE = 2


class _SyntheticDataset(Dataset):
    def __init__(self, n):
        self.input_ids = torch.randint(0, 50256, (32,))
        self.audio = [torch.randn(T_AUDIO, 768) for _ in range(n)]
        self.labels = [i % NUM_CLASSES for i in range(n)]

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids,
            "audio": self.audio[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main():
    device = "cpu"

    train_loader = DataLoader(_SyntheticDataset(8), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(_SyntheticDataset(4), batch_size=BATCH_SIZE, shuffle=False)

    compressor = AudioCompressor(target_len=50).to(device)
    model = AudioGPT2(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1,
        num_training_steps=EPOCHS * len(train_loader),
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, compressor(audio))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, compressor(audio))
                loss = criterion(logits, labels)
                val_loss += loss.item()
                all_preds.extend(logits.argmax(dim=-1).tolist())
                all_labels.extend(labels.tolist())

        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        print(f"Epoch {epoch}/{EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_f1: {val_f1:.4f}")

        assert torch.isfinite(torch.tensor(train_loss)), "train loss is NaN/Inf"
        assert torch.isfinite(torch.tensor(val_loss)), "val loss is NaN/Inf"

    print("\n✓ Training loop dry-run passed")


if __name__ == "__main__":
    main()
