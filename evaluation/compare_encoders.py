"""
Compare the full AudioGPT2 pipeline across all trained encoder checkpoints.

Run from project root after training each encoder:
    python evaluation/compare_encoders.py

Outputs (saved to checkpoints/):
    encoder_comparison.png  — grouped bar chart of accuracy and weighted F1
    best_encoder_cm.png     — confusion matrix for the best encoder by F1
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix

from data.dataset import EmoDBFusionDataset, speaker_independent_split
from models.compression.compressor import AudioCompressor
from models.audio_gpt2 import AudioGPT2

ENCODERS = ["wav2vec2-base", "wav2vec2-large-emotion", "wavlm-large", "hubert-large"]
ENCODER_LABELS = ["wav2vec2\nbase", "wav2vec2\nlarge-emotion", "wavlm\nlarge", "hubert\nlarge"]

VAL_SPEAKERS  = ["09", "10"]
TEST_SPEAKERS = ["03", "08"]
TARGET_AUDIO_LEN = 50
BATCH_SIZE = 8
OUTPUT_DIR = "checkpoints"


def _evaluate_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = checkpoint["encoder"]
    idx2label = checkpoint["idx2label"]
    num_classes = len(idx2label)

    embeddings_path = f"embeddings/{encoder}_embeddings.pt"
    dataset = EmoDBFusionDataset(embeddings_path)
    _, _, test_idx = speaker_independent_split(
        dataset, val_speakers=VAL_SPEAKERS, test_speakers=TEST_SPEAKERS
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False
    )

    audio_dim = dataset.embeddings[0].shape[-1]
    compressor = AudioCompressor(target_len=TARGET_AUDIO_LEN).to(device)
    model = AudioGPT2(num_classes=num_classes, audio_dim=audio_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, compressor(audio))
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    label_names = [idx2label[i] for i in range(num_classes)]

    return {"accuracy": acc, "f1": f1, "cm": cm, "label_names": label_names, "encoder": encoder}


def _plot_bar_chart(results, output_path):
    encoders_present = [r["encoder"] for r in results]
    labels = [ENCODER_LABELS[ENCODERS.index(e)] for e in encoders_present]
    accs = [r["accuracy"] for r in results]
    f1s  = [r["f1"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_acc = ax.bar(x - width / 2, accs, width, label="Accuracy")
    bars_f1  = ax.bar(x + width / 2, f1s,  width, label="Weighted F1")

    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Encoder comparison — full AudioGPT2 pipeline (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart → {output_path}")


def _plot_confusion_matrix(result, output_path):
    cm = result["cm"]
    label_names = result["label_names"]
    encoder = result["encoder"]

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {encoder} (normalized, test set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for encoder in ENCODERS:
        checkpoint_path = os.path.join(OUTPUT_DIR, f"{encoder}_best.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {encoder} — checkpoint not found at {checkpoint_path}")
            continue
        print(f"\nEvaluating {encoder}...")
        result = _evaluate_checkpoint(checkpoint_path, device)
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}  |  Weighted F1: {result['f1']:.4f}")

    if not results:
        print("No checkpoints found. Train at least one encoder first.")
        return

    _plot_bar_chart(results, os.path.join(OUTPUT_DIR, "encoder_comparison.png"))

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest encoder by F1: {best['encoder']} ({best['f1']:.4f})")
    _plot_confusion_matrix(best, os.path.join(OUTPUT_DIR, "best_encoder_cm.png"))

    print("\n── Summary ──────────────────────────────────────────")
    print(f"  {'Encoder':<30} {'Accuracy':>8}   {'W-F1':>6}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['encoder']:<30} {r['accuracy']*100:>7.1f}%   {r['f1']*100:>5.1f}%")


if __name__ == "__main__":
    main()
