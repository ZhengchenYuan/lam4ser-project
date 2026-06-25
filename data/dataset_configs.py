EMODB_LABELS = [
    "anger",
    "boredom",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
]

AIBO_LABELS = [
    "anger",
    "emphatic",
    "neutral",
    "positive",
    "rest",
]

DATASET_CONFIGS = {
    "emodb": {
        "labels": EMODB_LABELS,
        "embeddings_prefix": "",
        "checkpoint_dir": "checkpoints",
        "val_speakers": ["09", "10"],
        "test_speakers": ["03", "08"],
        "preprocessing_script": "models/audio_encoder/preprocessing.py",
    },
    "aibo": {
        "labels": AIBO_LABELS,
        "embeddings_prefix": "aibo_",
        "checkpoint_dir": "checkpoints_AIBO",
        "val_speakers": ["Ohm_31", "Ohm_32"],
        "test_speakers": [f"Mont_{i:02d}" for i in range(1, 26)],
        "preprocessing_script": "models/audio_encoder/preprocessing_aibo.py",
    },
}


def get_dataset_config(dataset: str) -> dict:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available datasets: {list(DATASET_CONFIGS.keys())}"
        )

    return DATASET_CONFIGS[dataset]
