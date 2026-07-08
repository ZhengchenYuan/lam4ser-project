import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from data.dataset import speaker_independent_split
from data.dataset_configs import DATASET_CONFIGS, get_dataset_config
from data.generation_dataset import EmoDBGenerationDataset, extract_speaker_id
from models.compression.compressor import AudioCompressor


CUE_LABELS = {
    "pitch": ["higher", "lower", "similar"],
    "energy": ["higher", "lower", "similar"],
    "rhythm": ["faster", "slower", "similar"],
    "duration": ["longer", "shorter", "similar"],
}
CUE_NAMES = tuple(CUE_LABELS.keys())
CUE_FEATURE_KEYS = ("pitch_mean", "energy_mean", "tempo", "duration")
BASELINE_ESTIMATION_MODES = ("speaker_neutral", "mixed_effects")


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
        self.acoustic_feature_vectors = []
        self.baseline_feature_vectors = []
        self.baseline_std_vectors = []

        for idx in range(len(self.base)):
            self._append_feature_vectors(idx)

    def _append_feature_vectors(self, idx: int):
        real_idx = self.base.sample_indices[idx]
        features = self.base.acoustic_feature_cache[real_idx]
        speaker_id = self.base.speaker_ids[idx]
        baseline = self.base.speaker_baselines.get(speaker_id, {})
        self.acoustic_feature_vectors.append(
            self._feature_vector_from_sample(features)
        )
        self.baseline_feature_vectors.append(
            self._feature_vector_from_baseline(baseline, "mean")
        )
        self.baseline_std_vectors.append(
            self._feature_vector_from_baseline(baseline, "std", default=1.0)
        )

    def rebuild_acoustic_cue_annotations(self):
        self.cue_targets = [
            self.base.build_acoustic_cue_target_for_sample(idx)
            for idx in range(len(self.base))
        ]
        self.acoustic_feature_vectors = []
        self.baseline_feature_vectors = []
        self.baseline_std_vectors = []
        for idx in range(len(self.base)):
            self._append_feature_vectors(idx)

    def _feature_vector_from_sample(self, features):
        return torch.tensor(
            [float(features.get(key, 0.0) or 0.0) for key in CUE_FEATURE_KEYS],
            dtype=torch.float,
        )

    def _feature_vector_from_baseline(
        self,
        baseline,
        stat_name: str,
        default: float = 0.0,
    ):
        values = []
        for key in CUE_FEATURE_KEYS:
            stats = baseline.get(key, {})
            values.append(float(stats.get(stat_name, default) or default))
        return torch.tensor(values, dtype=torch.float)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        base_item = self.base[idx]
        cue_target = self.cue_targets[idx]
        item = {
            "audio": base_item["audio"],
            "acoustic_features": self.acoustic_feature_vectors[idx],
            "baseline_features": self.baseline_feature_vectors[idx],
            "baseline_stds": self.baseline_std_vectors[idx],
        }
        for cue_name in CUE_NAMES:
            item[cue_name] = torch.tensor(
                self.cue_to_idx[cue_name][cue_target[cue_name]],
                dtype=torch.long,
            )
        return item


class TrainableBaselineAdapter(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, baseline_features):
        return self.scale * baseline_features + self.bias


class AcousticCueClassifier(nn.Module):
    def __init__(
        self,
        audio_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        trainable_baseline_adapter: bool = False,
        acoustic_feature_dim: int = len(CUE_FEATURE_KEYS),
    ):
        super().__init__()
        self.trainable_baseline_adapter = trainable_baseline_adapter
        self.encoder = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        head_input_dim = hidden_dim
        if trainable_baseline_adapter:
            self.baseline_adapter = TrainableBaselineAdapter(acoustic_feature_dim)
            self.feature_encoder = nn.Sequential(
                nn.LayerNorm(acoustic_feature_dim),
                nn.Linear(acoustic_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_input_dim += hidden_dim
        else:
            self.baseline_adapter = None
            self.feature_encoder = None

        self.heads = nn.ModuleDict({
            cue_name: nn.Linear(head_input_dim, len(CUE_LABELS[cue_name]))
            for cue_name in CUE_NAMES
        })

    def forward(
        self,
        audio_hidden,
        acoustic_features=None,
        baseline_features=None,
        baseline_stds=None,
    ):
        pooled = audio_hidden.mean(dim=1)
        encoded = self.encoder(pooled)

        if self.trainable_baseline_adapter:
            if acoustic_features is None or baseline_features is None:
                raise ValueError(
                    "Trainable baseline adapter requires acoustic and baseline features."
                )
            adapted_baseline = self.baseline_adapter(baseline_features)
            if baseline_stds is None:
                baseline_stds = torch.ones_like(adapted_baseline)
            relative_features = (
                acoustic_features - adapted_baseline
            ) / baseline_stds.clamp_min(1e-6)
            encoded = torch.cat(
                [encoded, self.feature_encoder(relative_features)],
                dim=-1,
            )

        return {
            cue_name: head(encoded)
            for cue_name, head in self.heads.items()
        }


def _checkpoint_path(args, dataset_config: dict) -> str:
    tag = f"{args.encoder}_acoustic_cue_classifier_{args.speaker_baseline_mode}"
    if args.baseline_estimation_mode == "mixed_effects":
        tag += "_mixed_effects"
    if args.trainable_baseline_adapter:
        tag += "_trainable_baseline"
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
        "baseline_estimation_mode": args.baseline_estimation_mode,
        "trainable_baseline_adapter": args.trainable_baseline_adapter,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _label_text(dataset: AcousticCueDataset, sample_idx: int) -> str:
    real_idx = dataset.base.sample_indices[sample_idx]
    return dataset.base._label_to_text(dataset.base.labels[real_idx])


def _neutral_enrollment_by_speaker(dataset: AcousticCueDataset):
    enrollment = {}
    for real_idx in sorted(dataset.base.enrollment_indices):
        label_text = dataset.base._label_to_text(dataset.base.labels[real_idx])
        if label_text != "neutral":
            continue
        speaker_id = extract_speaker_id(dataset.base.all_file_paths[real_idx])
        enrollment.setdefault(speaker_id, []).append(
            dataset.base.acoustic_feature_cache[real_idx]
        )
    return enrollment


def _speaker_id_from_real_idx(dataset: AcousticCueDataset, real_idx: int) -> str:
    return extract_speaker_id(dataset.base.all_file_paths[real_idx])


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = _mean(values)
    return sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)


def _estimate_mixed_effects_parameters(dataset: AcousticCueDataset, train_idx):
    neutral_by_speaker = {}
    for sample_idx in train_idx:
        if _label_text(dataset, sample_idx) != "neutral":
            continue
        real_idx = dataset.base.sample_indices[sample_idx]
        speaker_id = _speaker_id_from_real_idx(dataset, real_idx)
        features = dataset.base.acoustic_feature_cache[real_idx]
        neutral_by_speaker.setdefault(speaker_id, []).append(features)

    train_neutral_count = sum(len(features) for features in neutral_by_speaker.values())
    if train_neutral_count == 0:
        raise ValueError(
            "mixed_effects baseline estimation requires at least one train-split "
            "neutral sample."
        )

    parameters = {}
    for key in CUE_FEATURE_KEYS:
        values_by_speaker = {
            speaker_id: [
                float(features.get(key, 0.0) or 0.0)
                for features in speaker_features
            ]
            for speaker_id, speaker_features in neutral_by_speaker.items()
        }
        all_values = [
            value
            for speaker_values in values_by_speaker.values()
            for value in speaker_values
        ]
        global_mean = _mean(all_values)
        speaker_means = {
            speaker_id: _mean(speaker_values)
            for speaker_id, speaker_values in values_by_speaker.items()
        }
        residual_sse = sum(
            (value - speaker_means[speaker_id]) ** 2
            for speaker_id, speaker_values in values_by_speaker.items()
            for value in speaker_values
        )
        residual_df = train_neutral_count - len(values_by_speaker)
        residual_variance = residual_sse / residual_df if residual_df > 0 else 0.0
        mean_variance = _variance(list(speaker_means.values()))
        mean_inverse_n = _mean([
            1.0 / len(speaker_values)
            for speaker_values in values_by_speaker.values()
        ])
        speaker_variance = max(
            0.0,
            mean_variance - residual_variance * mean_inverse_n,
        )
        parameters[key] = {
            "mu": global_mean,
            "speaker_variance": speaker_variance,
            "residual_variance": residual_variance,
        }

    return parameters, train_neutral_count


def apply_mixed_effects_baselines(dataset: AcousticCueDataset, train_idx) -> dict:
    parameters, train_neutral_count = _estimate_mixed_effects_parameters(
        dataset,
        train_idx,
    )
    neutral_enrollment = _neutral_enrollment_by_speaker(dataset)
    target_speakers = sorted(set(dataset.base.speaker_ids))
    fallback_speakers = []

    for speaker_id in target_speakers:
        speaker_features = neutral_enrollment.get(speaker_id, [])
        if not speaker_features:
            fallback_speakers.append(speaker_id)

        baseline = {
            key: dict(value)
            for key, value in dataset.base.speaker_baselines.get(speaker_id, {}).items()
        }
        for key, stats in parameters.items():
            feature_stats = baseline.setdefault(key, {"mean": stats["mu"], "std": 1.0})
            if not speaker_features:
                feature_stats["mean"] = stats["mu"]
                continue

            speaker_values = [
                float(features.get(key, 0.0) or 0.0)
                for features in speaker_features
            ]
            speaker_mean = _mean(speaker_values)
            n_s = len(speaker_values)
            denominator = (
                stats["speaker_variance"]
                + stats["residual_variance"] / max(n_s, 1)
            )
            lambda_sf = (
                stats["speaker_variance"] / denominator
                if denominator > 0.0
                else 0.0
            )
            feature_stats["mean"] = (
                stats["mu"] + lambda_sf * (speaker_mean - stats["mu"])
            )

        dataset.base.speaker_baselines[speaker_id] = baseline

    dataset.rebuild_acoustic_cue_annotations()
    return {
        "mode": "mixed_effects",
        "train_neutral_samples": train_neutral_count,
        "parameters": parameters,
        "fallback_speakers": fallback_speakers,
        "fallback_speaker_count": len(fallback_speakers),
    }


def apply_baseline_estimation(dataset: AcousticCueDataset, train_idx, config) -> dict:
    mode = config["baseline_estimation_mode"]
    if mode == "speaker_neutral":
        return {
            "mode": "speaker_neutral",
            "train_neutral_samples": None,
            "parameters": {},
            "fallback_speakers": [],
            "fallback_speaker_count": 0,
        }
    if mode == "mixed_effects":
        return apply_mixed_effects_baselines(dataset, train_idx)
    raise ValueError(f"Unknown baseline_estimation_mode: {mode}")


def print_baseline_estimation_summary(summary: dict):
    print(f"  Baseline estimation mode: {summary['mode']}")
    if summary["mode"] != "mixed_effects":
        return

    print(f"  Train neutral samples: {summary['train_neutral_samples']}")
    print(f"  Mixed-effects fallback speakers: {summary['fallback_speaker_count']}")
    if summary["fallback_speakers"]:
        print(f"    {summary['fallback_speakers']}")
    print("  Mixed-effects parameters:")
    for key in CUE_FEATURE_KEYS:
        stats = summary["parameters"][key]
        print(
            f"    {key}: "
            f"mu={stats['mu']:.6f}, "
            f"sigma_speaker^2={stats['speaker_variance']:.6f}, "
            f"sigma_residual^2={stats['residual_variance']:.6f}"
        )


def _cue_loss(logits_by_cue, batch, criterion) -> torch.Tensor:
    losses = [
        criterion(logits_by_cue[cue_name], batch[cue_name])
        for cue_name in CUE_NAMES
    ]
    return sum(losses) / len(losses)


def _batch_adapter_inputs(batch, device, enabled: bool):
    if not enabled:
        return {}

    return {
        "acoustic_features": batch["acoustic_features"].to(device),
        "baseline_features": batch["baseline_features"].to(device),
        "baseline_stds": batch["baseline_stds"].to(device),
    }


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
            logits_by_cue = model(
                audio_compressed,
                **_batch_adapter_inputs(
                    batch,
                    device,
                    model.trainable_baseline_adapter,
                ),
            )
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
    baseline_summary = apply_baseline_estimation(dataset, train_idx, config)
    config["baseline_estimation_summary"] = baseline_summary

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
        trainable_baseline_adapter=config["trainable_baseline_adapter"],
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
    print_baseline_estimation_summary(baseline_summary)
    print(f"  Trainable baseline adapter: {config['trainable_baseline_adapter']}")
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
            logits_by_cue = model(
                audio_compressed,
                **_batch_adapter_inputs(
                    batch,
                    device,
                    model.trainable_baseline_adapter,
                ),
            )
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
                    "baseline_estimation_mode": config["baseline_estimation_mode"],
                    "baseline_estimation_summary": (
                        config["baseline_estimation_summary"]
                    ),
                    "trainable_baseline_adapter": (
                        config["trainable_baseline_adapter"]
                    ),
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
    parser.add_argument(
        "--baseline_estimation_mode",
        choices=BASELINE_ESTIMATION_MODES,
        default="speaker_neutral",
        help=(
            "Statistical speaker-baseline estimation mode. speaker_neutral "
            "reproduces Q3; mixed_effects applies train-split neutral "
            "random-intercept partial pooling before classifier training."
        ),
    )
    parser.add_argument(
        "--trainable_baseline_adapter",
        action="store_true",
        help=(
            "Enable a shared trainable adapter for speaker baseline acoustic "
            "feature means."
        ),
    )
    parser.add_argument("--checkpoint_path", default=None)

    args = parser.parse_args()
    train(_build_config(args))
