import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset

from data.dataset import speaker_independent_split
from data.dataset_configs import DATASET_CONFIGS, get_dataset_config
from models.compression.compressor import AudioCompressor
from training.train_acoustic_cue_classifier import (
    CUE_LABELS,
    CUE_NAMES,
    AcousticCueClassifier,
    AcousticCueDataset,
)


def _build_config(args) -> dict:
    dataset_config = get_dataset_config(args.dataset)
    tag = f"{args.encoder}_acoustic_cue_classifier"

    return {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "embeddings_path": (
            f"embeddings/{dataset_config['embeddings_prefix']}"
            f"{args.encoder}_embeddings.pt"
        ),
        "checkpoint_path": (
            args.checkpoint_path
            or f"{dataset_config['checkpoint_dir']}/{tag}_best.pt"
        ),
        "preprocessing_script": dataset_config["preprocessing_script"],
        "val_speakers": dataset_config["val_speakers"],
        "test_speakers": dataset_config["test_speakers"],
        "batch_size": args.batch_size,
        "target_audio_len": args.target_audio_len,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _evaluate(model, compressor, loader, device):
    model.eval()
    y_true = {cue_name: [] for cue_name in CUE_NAMES}
    y_pred = {cue_name: [] for cue_name in CUE_NAMES}

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            audio_compressed = compressor(audio)
            logits_by_cue = model(audio_compressed)

            for cue_name in CUE_NAMES:
                preds = logits_by_cue[cue_name].argmax(dim=-1)
                y_pred[cue_name].extend(preds.cpu().tolist())
                y_true[cue_name].extend(batch[cue_name].tolist())

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
        "cue_accuracies": cue_accuracies,
        "macro_cue_accuracy": sum(cue_accuracies.values()) / len(cue_accuracies),
        "exact_all_cue_match": exact_matches / sample_count,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate(config):
    if not os.path.exists(config["embeddings_path"]):
        raise FileNotFoundError(
            f"Embeddings file not found: {config['embeddings_path']}. "
            f"Run {config['preprocessing_script']} first."
        )
    if not os.path.exists(config["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint_path']}")

    device = config["device"]
    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )
    checkpoint_config = checkpoint.get("config", {})
    hidden_dim = checkpoint_config.get("hidden_dim", config["hidden_dim"])
    dropout = checkpoint_config.get("dropout", config["dropout"])
    target_audio_len = checkpoint_config.get(
        "target_audio_len",
        config["target_audio_len"],
    )

    dataset = AcousticCueDataset(config["embeddings_path"])
    _, _, test_idx = speaker_independent_split(
        dataset,
        val_speakers=config["val_speakers"],
        test_speakers=config["test_speakers"],
    )
    loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    audio_dim = dataset[0]["audio"].shape[-1]
    compressor = AudioCompressor(target_len=target_audio_len).to(device)
    model = AcousticCueClassifier(
        audio_dim=audio_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = _evaluate(model, compressor, loader, device)

    print("\nAcoustic cue classifier evaluation configuration:")
    print(f"  Dataset:     {config['dataset']}")
    print(f"  Encoder:     {config['encoder']}")
    print(f"  Device:      {device}")
    print(f"  Checkpoint:  {config['checkpoint_path']}")
    print()

    print("Acoustic cue classifier results:")
    print(f"  Pitch accuracy:        {metrics['cue_accuracies']['pitch']:.4f}")
    print(f"  Energy accuracy:       {metrics['cue_accuracies']['energy']:.4f}")
    print(f"  Rhythm accuracy:       {metrics['cue_accuracies']['rhythm']:.4f}")
    print(f"  Duration accuracy:     {metrics['cue_accuracies']['duration']:.4f}")
    print(f"  Macro cue accuracy:    {metrics['macro_cue_accuracy']:.4f}")
    print(f"  Exact all-cue match:   {metrics['exact_all_cue_match']:.4f}")

    print("\nConfusion matrices (rows=true, cols=pred):")
    for cue_name in CUE_NAMES:
        labels = list(range(len(CUE_LABELS[cue_name])))
        cm = confusion_matrix(
            metrics["y_true"][cue_name],
            metrics["y_pred"][cue_name],
            labels=labels,
        )
        print(f"\n{cue_name}: {CUE_LABELS[cue_name]}")
        print(cm)


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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--target_audio_len", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--checkpoint_path", default=None)

    args = parser.parse_args()
    evaluate(_build_config(args))
