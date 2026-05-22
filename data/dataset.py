import os
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

_PROMPT = "Classify the emotion of this speech:"


def extract_speaker_id(file_path: str) -> str:
    basename = os.path.basename(file_path)
    if len(basename) < 2:
        return "unknown"
    return basename[:2]


class EmoDBFusionDataset(Dataset):
    def __init__(self, embeddings_path: str):
        if not os.path.exists(embeddings_path):
            print(
                f"ERROR: '{embeddings_path}' not found. "
                "Run models/fusion/preprocessing.py first to generate the embeddings file."
            )
            raise FileNotFoundError(f"'{embeddings_path}' not found")

        data = torch.load(embeddings_path, weights_only=False)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.label2idx = data["label2idx"]
        self.idx2label = data["idx2label"]

        self.speaker_ids = None
        for key in ("file_paths", "paths", "files"):
            if key in data:
                self.speaker_ids = [extract_speaker_id(p) for p in data[key]]
                break

        if self.speaker_ids is None:
            print(
                "WARNING: No file paths found in emodb_embeddings.pt.\n"
                "Speaker-independent splitting is not available.\n"
                "Falling back to random 70/15/15 split."
            )

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(
            _PROMPT,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"].squeeze(0)  # [32]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids,
            "audio": self.embeddings[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def speaker_independent_split(dataset, val_speakers=None, test_speakers=None):
    if test_speakers is None:
        test_speakers = ["03", "08"]
    if val_speakers is None:
        val_speakers = ["09", "10"]

    if dataset.speaker_ids is None:
        torch.manual_seed(42)
        n = len(dataset)
        indices = torch.randperm(n).tolist()
        train_end = int(0.70 * n)
        val_end = train_end + int(0.15 * n)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        print("Random 70/15/15 split:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val:   {len(val_indices)} samples")
        print(f"  Test:  {len(test_indices)} samples")

        if not train_indices or not val_indices or not test_indices:
            raise ValueError("One or more splits are empty after random 70/15/15 split.")

        return train_indices, val_indices, test_indices

    test_speakers = set(test_speakers)
    val_speakers = set(val_speakers)
    train_indices, val_indices, test_indices = [], [], []

    for i, spk in enumerate(dataset.speaker_ids):
        if spk in test_speakers:
            test_indices.append(i)
        elif spk in val_speakers:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_speakers = sorted(set(dataset.speaker_ids[i] for i in train_indices))
    print("Speaker split summary:")
    print(f"  Train speakers: {train_speakers} → {len(train_indices)} samples")
    print(f"  Val   speakers: {sorted(val_speakers)} → {len(val_indices)} samples")
    print(f"  Test  speakers: {sorted(test_speakers)} → {len(test_indices)} samples")

    if not train_indices:
        raise ValueError("Train split is empty — check speaker IDs in the dataset.")
    if not val_indices:
        raise ValueError("Val split is empty — check val_speakers argument.")
    if not test_indices:
        raise ValueError("Test split is empty — check test_speakers argument.")

    return train_indices, val_indices, test_indices
