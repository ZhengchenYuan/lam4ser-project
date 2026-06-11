"""
SVM baseline for AIBO emotion recognition (IS2009 Emotion Challenge, 5-class).

Features: 40 MFCCs (mean + std), F0 (mean + std of voiced frames), RMS energy (mean + std).
Uses the same speaker-independent split as the main model.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import audiofile
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data.dataset import extract_speaker_id

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# EMoDB: WAV_DIR = os.path.join(PROJECT_ROOT, "audb", "emodb", "2.0.0", "fe182b91", "wav")
WAV_DIR = os.path.join(PROJECT_ROOT, "dataset", "wav")
LABELS_FILE = os.path.join(
    PROJECT_ROOT, "dataset", "labels", "IS2009EmotionChallenge", "chunk_labels_5cl_corpus.txt"
)

# EMoDB: VAL_SPEAKERS = {"09", "10"}, TEST_SPEAKERS = {"03", "08"}
VAL_SPEAKERS  = {"Ohm_31", "Ohm_32"}
TEST_SPEAKERS = {f"Mont_{i:02d}" for i in range(1, 26)}
N_MFCC        = 40
SAMPLING_RATE = 16000

# EMoDB:
# _EMOTION_CODES = {
#     "W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
#     "F": "happiness", "N": "neutral", "T": "sadness",
# }

# AIBO (IS2009 5-class codes):
AIBO_LABEL_MAP = {
    "A": "anger", "E": "emphatic", "N": "neutral", "P": "positive", "R": "rest",
}


def _load_label_index():
    """Read the IS2009 5-class label file -> list of (wav_path, emotion, speaker_id)."""
    index = []
    with open(LABELS_FILE) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            chunk_name, code = parts[0], parts[1]
            wav_path = os.path.join(WAV_DIR, f"{chunk_name}.wav")
            emotion = AIBO_LABEL_MAP[code]
            index.append((wav_path, emotion, extract_speaker_id(wav_path)))
    return index


def _extract_features(path):
    signal, sr = audiofile.read(path, always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=0)
    signal = signal.astype(np.float32)
    if sr != SAMPLING_RATE:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SAMPLING_RATE)

    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLING_RATE, n_mfcc=N_MFCC)
    mfcc_feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

    f0 = librosa.yin(signal, fmin=75.0, fmax=600.0, sr=SAMPLING_RATE)
    voiced = f0[f0 > 0]
    pitch_feat = np.array([voiced.mean(), voiced.std()]) if len(voiced) else np.zeros(2)

    rms = librosa.feature.rms(y=signal)[0]
    energy_feat = np.array([rms.mean(), rms.std()])

    return np.concatenate([mfcc_feat, pitch_feat, energy_feat])


def run():
    if not os.path.isdir(WAV_DIR):
        print(f"ERROR: wav directory not found at {WAV_DIR}")
        sys.exit(1)

    index = _load_label_index()
    print(f"Found {len(index)} labeled samples, extracting features...")

    emotion_classes = sorted(set(AIBO_LABEL_MAP.values()))
    label2idx = {e: i for i, e in enumerate(emotion_classes)}
    idx2label  = {i: e for e, i in label2idx.items()}

    features, labels, speaker_ids = [], [], []
    for i, (path, emotion, speaker_id) in enumerate(index):
        features.append(_extract_features(path))
        labels.append(label2idx[emotion])
        speaker_ids.append(speaker_id)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(index)}")

    features = np.array(features)
    labels   = np.array(labels)

    train_idx, val_idx, test_idx = [], [], []
    for i, spk in enumerate(speaker_ids):
        if spk in TEST_SPEAKERS:
            test_idx.append(i)
        elif spk in VAL_SPEAKERS:
            val_idx.append(i)
        else:
            train_idx.append(i)

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val,   y_val   = features[val_idx],   labels[val_idx]
    X_test,  y_test  = features[test_idx],  labels[test_idx]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    svm = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=42)
    svm.fit(X_train, y_train)

    val_preds = svm.predict(X_val)
    print(f"Val  — accuracy: {accuracy_score(y_val, val_preds):.4f}  "
          f"weighted F1: {f1_score(y_val, val_preds, average='weighted'):.4f}")

    test_preds  = svm.predict(X_test)
    test_acc    = accuracy_score(y_test, test_preds)
    test_f1     = f1_score(y_test, test_preds, average="weighted")
    cm          = confusion_matrix(y_test, test_preds)
    label_names = [idx2label[i] for i in range(len(idx2label))]

    print(f"\nSVM + MFCC + pitch + energy  —  test ({len(y_test)} samples)")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  Weighted F1: {test_f1:.4f}")
    print(classification_report(y_test, test_preds, target_names=label_names))
    print("  " + "  ".join(f"{n[:4]:>4}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i][:6]:<6}  {'  '.join(f'{v:4d}' for v in row)}")

    return {"accuracy": test_acc, "f1": test_f1}


if __name__ == "__main__":
    run()
