import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from data.dataset import speaker_independent_split
from data.dataset_configs import DATASET_CONFIGS, get_dataset_config
from data.generation_dataset import EmoDBGenerationDataset
from models.compression.compressor import AudioCompressor


CUE_LABELS = {
    "pitch": ["higher", "lower", "similar"],
    "energy": ["higher", "lower", "similar"],
    "rhythm": ["faster", "slower", "similar"],
    "duration": ["longer", "shorter", "similar"],
}
CUE_NAMES = tuple(CUE_LABELS.keys())


class AcousticCueDataset(Dataset):
    def __init__(self, embeddings_path: str, speaker_baseline_mode: str = "neutral"):
        self.base = EmoDBGenerationDataset(
            embeddings_path=embeddings_path,
            prompt_type="speaker_acoustic_cue_generation",
            max_length=128,
            speaker_baseline_mode=speaker_baseline_mode,
        )
        self.speaker_ids = self.base.speaker_ids
        self.embeddings = self.base.embeddings
        self.speaker_baseline_mode = speaker_baseline_mode
        self.cue_to_idx = {
            cue_name: {label: idx for idx, label in enumerate(labels)}
            for cue_name, labels in CUE_LABELS.items()
        }
        self.idx_to_cue = {
            cue_name: {idx: label for idx, label in enumerate(labels)}
            for cue_name, labels in CUE_LABELS.items()
        }
        self.cue_targets = [
            self.base.build_acoustic_cue_target_for_sample(idx)
            for idx in range(len(self.base))
        ]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        base_item = self.base[idx]
        cue_target = self.cue_targets[idx]
        item = {"audio": base_item["audio"]}
        for cue_name in CUE_NAMES:
            item[cue_name] = torch.tensor(
                self.cue_to_idx[cue_name][cue_target[cue_name]],
                dtype=torch.long,
            )
        return item


class AcousticCueClassifier(nn.Module):
    def __init__(self, audio_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            cue_name: nn.Linear(hidden_dim, len(CUE_LABELS[cue_name]))
            for cue_name in CUE_NAMES
        })

    def forward(self, audio_hidden):
        pooled = audio_hidden.mean(dim=1)
        encoded = self.encoder(pooled)
        return {
            cue_name: head(encoded)
            for cue_name, head in self.heads.items()
        }


def _checkpoint_path(args, dataset_config: dict) -> str:
    tag = f"{args.encoder}_acoustic_cue_classifier_{args.speaker_baseline_mode}"
    return args.checkpoint_path or f"{dataset_config['checkpoint_dir']}/{tag}_best.pt"


def _build_config(args) -> dict:
    dataset_config = get_dataset_config(args.dataset)
    os.makedirs(dataset_config["checkpoint_dir"], exist_ok=True)

    return {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "speaker_baseline_mode": args.speaker_baseline_mode,
        "embeddings_path": (
            f"embeddings/{dataset_config['embeddings_prefix']}"
            f"{args.encoder}_embeddings.pt"
        ),
        "checkpoint_path": _checkpoint_path(args, dataset_config),
        "preprocessing_script": dataset_config["preprocessing_script"],
        "val_speakers": dataset_config["val_speakers"],
        "test_speakers": dataset_config["test_speakers"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "target_audio_len": args.target_audio_len,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _cue_loss(logits_by_cue, batch, criterion) -> torch.Tensor:
    losses = [
        criterion(logits_by_cue[cue_name], batch[cue_name])
        for cue_name in CUE_NAMES
    ]
    return sum(losses) / len(losses)


def _evaluate(model, compressor, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = {cue_name: [] for cue_name in CUE_NAMES}
    y_pred = {cue_name: [] for cue_name in CUE_NAMES}

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            targets = {
                cue_name: batch[cue_name].to(device)
                for cue_name in CUE_NAMES
            }
            audio_compressed = compressor(audio)
            logits_by_cue = model(audio_compressed)
            total_loss += _cue_loss(logits_by_cue, targets, criterion).item()

            for cue_name in CUE_NAMES:
                preds = logits_by_cue[cue_name].argmax(dim=-1)
                y_pred[cue_name].extend(preds.cpu().tolist())
                y_true[cue_name].extend(targets[cue_name].cpu().tolist())

    cue_accuracies = {}
    for cue_name in CUE_NAMES:
        correct = sum(
            pred == true
            for pred, true in zip(y_pred[cue_name], y_true[cue_name])
        )
        cue_accuracies[cue_name] = correct / max(len(y_true[cue_name]), 1)

    sample_count = max(len(y_true[CUE_NAMES[0]]), 1)
    exact_matches = 0
    for idx in range(sample_count):
        if all(y_pred[cue_name][idx] == y_true[cue_name][idx] for cue_name in CUE_NAMES):
            exact_matches += 1

    return {
        "loss": total_loss / max(len(loader), 1),
        "cue_accuracies": cue_accuracies,
        "macro_cue_accuracy": sum(cue_accuracies.values()) / len(cue_accuracies),
        "exact_all_cue_match": exact_matches / sample_count,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train(config):
    if not os.path.exists(config["embeddings_path"]):
        print(
            f"ERROR: '{config['embeddings_path']}' not found. "
            f"Run {config['preprocessing_script']} first to generate embeddings."
        )
        sys.exit(1)

    dataset = AcousticCueDataset(
        config["embeddings_path"],
        speaker_baseline_mode=config["speaker_baseline_mode"],
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
    audio_dim = dataset[0]["audio"].shape[-1]
    compressor = AudioCompressor(target_len=config["target_audio_len"]).to(device)
    model = AcousticCueClassifier(
        audio_dim=audio_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-2,
    )

    print("\nAcoustic cue classifier training configuration:")
    print(f"  Dataset:      {config['dataset']}")
    print(f"  Encoder:      {config['encoder']}")
    print(f"  Baseline mode:{config['speaker_baseline_mode']}")
    print(f"  Device:       {device}")
    print(f"  Checkpoint:   {config['checkpoint_path']}")
    print(f"  Cue labels:   {CUE_LABELS}")
    print()

    best_val_macro = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            audio = batch["audio"].to(device)
            targets = {
                cue_name: batch[cue_name].to(device)
                for cue_name in CUE_NAMES
            }

            audio_compressed = compressor(audio)
            logits_by_cue = model(audio_compressed)
            loss = _cue_loss(logits_by_cue, targets, criterion)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_metrics = _evaluate(model, compressor, val_loader, criterion, device)

        print(
            f"Epoch {epoch:2d}/{config['epochs']} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_macro_cue_acc: {val_metrics['macro_cue_accuracy']:.4f} | "
            f"val_exact: {val_metrics['exact_all_cue_match']:.4f}"
        )

        is_best = (
            val_metrics["macro_cue_accuracy"] > best_val_macro
            or (
                val_metrics["macro_cue_accuracy"] == best_val_macro
                and val_metrics["loss"] < best_val_loss
            )
        )
        if is_best:
            best_val_macro = val_metrics["macro_cue_accuracy"]
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "cue_labels": CUE_LABELS,
                    "speaker_baseline_mode": config["speaker_baseline_mode"],
                    "val_loss": val_metrics["loss"],
                    "val_macro_cue_accuracy": val_metrics["macro_cue_accuracy"],
                    "val_exact_all_cue_match": val_metrics["exact_all_cue_match"],
                },
                config["checkpoint_path"],
            )
            print(
                "  Saved best checkpoint "
                f"(val_macro_cue_acc: {best_val_macro:.4f})"
            )

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _evaluate(model, compressor, test_loader, criterion, device)

    print(f"\nTest results (best checkpoint, epoch {checkpoint['epoch']}):")
    print(f"  Pitch accuracy:        {test_metrics['cue_accuracies']['pitch']:.4f}")
    print(f"  Energy accuracy:       {test_metrics['cue_accuracies']['energy']:.4f}")
    print(f"  Rhythm accuracy:       {test_metrics['cue_accuracies']['rhythm']:.4f}")
    print(f"  Duration accuracy:     {test_metrics['cue_accuracies']['duration']:.4f}")
    print(f"  Macro cue accuracy:    {test_metrics['macro_cue_accuracy']:.4f}")
    print(f"  Exact all-cue match:   {test_metrics['exact_all_cue_match']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="emodb", choices=list(DATASET_CONFIGS))
    parser.add_argument(
        "--encoder",
        default="wavlm-large",
        choices=[
            "wav2vec2-base",
            "wav2vec2-large-emotion",
            "wavlm-large",
            "hubert-large",
        ],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--target_audio_len", type=int, default=50)
    parser.add_argument(
        "--speaker_baseline_mode",
        choices=["neutral", "emotion_balanced"],
        default="neutral",
        help="Speaker baseline mode used to derive acoustic cue labels.",
    )
    parser.add_argument("--checkpoint_path", default=None)

    args = parser.parse_args()
    train(_build_config(args))
