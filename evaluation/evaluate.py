"""
Evaluate the best saved checkpoint on the held-out test set.
Run from the project root: python evaluation/evaluate.py
"""
import sys
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from data.dataset import EmoDBFusionDataset, speaker_independent_split
from models.compression.compressor import AudioCompressor
from models.audio_gpt2 import AudioGPT2

CONFIG = {
    # EMoDB: "embeddings_path": "embeddings/emodb_embeddings.pt"
    # EMoDB: "val_speakers": ["09", "10"], "test_speakers": ["03", "08"]
    "embeddings_path": "embeddings/aibo_wavlm-large_embeddings.pt",
    "checkpoint_path": "best_model.pt",
    "target_audio_len": 50,
    "batch_size": 8,
    "val_speakers": ["Ohm_31", "Ohm_32"],
    "test_speakers": [f"Mont_{i:02d}" for i in range(1, 26)],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def evaluate():
    for path in (CONFIG["embeddings_path"], CONFIG["checkpoint_path"]):
        if not __import__("os").path.exists(path):
            print(f"ERROR: '{path}' not found.")
            if path == CONFIG["embeddings_path"]:
                print("  Run models/audio_encoder/preprocessing_aibo.py first.")
            else:
                print("  Run training/train_base_model.py first.")
            sys.exit(1)

    device = CONFIG["device"]

    dataset = EmoDBFusionDataset(CONFIG["embeddings_path"])
    _, _, test_idx = speaker_independent_split(
        dataset,
        val_speakers=CONFIG["val_speakers"],
        test_speakers=CONFIG["test_speakers"],
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=CONFIG["batch_size"], shuffle=False
    )

    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=False)
    idx2label = checkpoint["idx2label"]
    num_classes = len(idx2label)

    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}  |  "
          f"Val acc: {checkpoint['val_acc']:.4f}  |  "
          f"Val F1: {checkpoint['val_f1']:.4f}")

    compressor = AudioCompressor(target_len=CONFIG["target_audio_len"]).to(device)
    model = AudioGPT2(num_classes=num_classes).to(device)
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

    label_names = [idx2label[i] for i in range(num_classes)]
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n── Test set results ({len(all_labels)} samples) ──────────────────")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Weighted F1: {f1:.4f}")
    print(f"\n── Per-class report ────────────────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    print(f"── Confusion matrix (rows=true, cols=pred) ─────────────────")
    header = "        " + "  ".join(f"{n[:4]:>4}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:4d}" for v in row)
        print(f"  {label_names[i][:6]:<6}  {row_str}")


if __name__ == "__main__":
    evaluate()
