"""
Linear and MLP probes on mean-pooled audio embeddings.

Loads a saved embeddings .pt file, mean-pools the variable-length sequences to a
fixed-size vector, and trains two simple classifiers. Both use the same
speaker-independent split as the main model.

How to run:
python baselines/embedding_probes.py --embeddings embeddings/hubert_embeddings.pt
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from data.dataset import extract_speaker_id

# Default path
EMBEDDINGS_PATH = "embeddings/aibo_wavlm-large_embeddings.pt"

# EMoDB: VAL_SPEAKERS = {"09", "10"}, TEST_SPEAKERS = {"03", "08"}
VAL_SPEAKERS  = {"Ohm_31", "Ohm_32"}
TEST_SPEAKERS = {f"Mont_{i:02d}" for i in range(1, 26)}

# Just linear
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Small MLP
class MLPProbe(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _load_and_pool(embeddings_path):
    data = torch.load(embeddings_path, weights_only=False)
    pooled = torch.stack([emb.mean(dim=0) for emb in data["embeddings"]])
    labels = torch.tensor(data["labels"], dtype=torch.long)

    file_paths = data.get("file_paths") or data.get("paths") or data.get("files")
    if file_paths is None:
        raise ValueError("No file_paths key in embeddings file; re-run preprocessing.py.")
    speaker_ids = [extract_speaker_id(p) for p in file_paths]

    return pooled, labels, speaker_ids, data["idx2label"]


def _speaker_split(speaker_ids):
    train_idx, val_idx, test_idx = [], [], []
    for i, spk in enumerate(speaker_ids):
        if spk in TEST_SPEAKERS:
            test_idx.append(i)
        elif spk in VAL_SPEAKERS:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx


def _train(model, X_train, y_train, X_val, y_val, class_weights, device, epochs=300, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loader = DataLoader(
        TensorDataset(X_train.to(device), y_train.to(device)),
        batch_size=32, shuffle=True,
    )

    best_val_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val.to(device)).argmax(dim=-1) == y_val.to(device)).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def _evaluate(name, model, X_test, y_test, idx2label, device):
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=-1).cpu().tolist()

    true        = y_test.tolist()
    acc         = sum(p == l for p, l in zip(preds, true)) / len(true)
    f1          = f1_score(true, preds, average="weighted")
    cm          = confusion_matrix(true, preds)
    label_names = [idx2label[i] for i in range(len(idx2label))]

    print(f"\n{name}  —  test ({len(true)} samples)")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Weighted F1: {f1:.4f}")
    print(classification_report(true, preds, target_names=label_names))
    print("  " + "  ".join(f"{n[:4]:>4}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i][:6]:<6}  {'  '.join(f'{v:4d}' for v in row)}")

    return {"accuracy": acc, "f1": f1}


def run(embeddings_path=EMBEDDINGS_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading embeddings from {embeddings_path}...")
    pooled, labels, speaker_ids, idx2label = _load_and_pool(embeddings_path)
    input_dim, num_classes = pooled.shape[1], len(idx2label)
    print(f"  {len(pooled)} samples, dim={input_dim}, {num_classes} classes")

    train_idx, val_idx, test_idx = _speaker_split(speaker_ids)
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    X_train, y_train = pooled[train_idx], labels[train_idx]
    X_val,   y_val   = pooled[val_idx],   labels[val_idx]
    X_test,  y_test  = pooled[test_idx],  labels[test_idx]

    cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train.numpy())
    class_weights = torch.tensor(cw, dtype=torch.float)

    encoder_tag = os.path.splitext(os.path.basename(embeddings_path))[0]
    results = {}

    print("\nTraining linear probe...")
    linear = _train(LinearProbe(input_dim, num_classes), X_train, y_train, X_val, y_val, class_weights, device)
    results["linear"] = _evaluate(f"Linear probe  ({encoder_tag})", linear, X_test, y_test, idx2label, device)

    print("\nTraining MLP probe...")
    mlp = _train(MLPProbe(input_dim, num_classes), X_train, y_train, X_val, y_val, class_weights, device)
    results["mlp"] = _evaluate(f"MLP probe  ({encoder_tag})", mlp, X_test, y_test, idx2label, device)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default=EMBEDDINGS_PATH)
    args = parser.parse_args()
    run(args.embeddings)
