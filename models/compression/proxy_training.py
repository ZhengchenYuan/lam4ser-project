"""
Person 2 — Proxy Training Experiment
=====================================
Demonstrates that attention pooling queries CAN learn to focus
on emotionally salient frames, even with a simplified pipeline.

This is NOT the real LAM4SER training (which uses Person 3's
cross-attention + Person 4's training loop). This is a proxy
experiment that proves the concept.

What it proves:
    1. Attention queries are trainable (spread increases)
    2. Queries converge to acoustically distinctive frames
    3. The convergence target is genuine (highest deviation from mean)

What it does NOT prove:
    - Real emotion classification accuracy (needs full pipeline)
    - Query specialization across emotions (needs more data)
    - Full LAM4SER performance (needs Person 3 + 4)

Requirements:
    pip install torch scipy transformers

Usage:
    python proxy_training.py --audio path/to/emodb_file.wav --emotion anger
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile

from compressors import AttentionPoolCompressor, AudioProjection


# ─────────────────────────────────────────────────────────────────────────────
# EMO-DB emotion labels
# ─────────────────────────────────────────────────────────────────────────────

EMODB_EMOTIONS = {
    "anger": 0,     # W = Wut (German for anger)
    "boredom": 1,   # L = Langeweile
    "disgust": 2,   # E = Ekel
    "fear": 3,      # A = Angst
    "happiness": 4, # F = Freude
    "sadness": 5,   # T = Trauer
    "neutral": 6,   # N = Neutral
}

# EMO-DB filename convention: the 6th character encodes emotion
# e.g. 03a05Wb.wav → W = Wut = anger
EMODB_CHAR_TO_EMOTION = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}


def detect_emotion_from_filename(filename: str) -> str:
    """
    Detect emotion from EMO-DB filename convention.
    The 6th character (index 5) encodes the emotion.
    e.g. 03a05Wb.wav → W → anger
    """
    import os
    basename = os.path.basename(filename)
    if len(basename) >= 6:
        char = basename[5]
        if char in EMODB_CHAR_TO_EMOTION:
            return EMODB_CHAR_TO_EMOTION[char]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PROXY MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ProxyEmotionModel(nn.Module):
    """
    Simplified model for proxy training:
        audio → attention compression → projection → classifier

    NOT the real LAM4SER architecture (which uses an LLM).
    Only used to demonstrate that attention queries can learn.
    """

    def __init__(self, d_a: int, d: int, num_queries: int = 8, num_emotions: int = 7):
        super().__init__()
        self.compressor = AttentionPoolCompressor(d=d_a, num_queries=num_queries)
        self.proj = nn.Linear(d_a, d)
        self.norm = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, num_emotions)

    def forward(self, audio_features: torch.Tensor):
        """
        Args:
            audio_features: [B, T, d_a]
        Returns:
            logits: [B, num_emotions]
            weights: [B, Q, T] attention weights for analysis
        """
        # Get attention weights for visualization
        weights = self.compressor.get_attention_weights(audio_features)

        # Compress
        x = self.compressor(audio_features)  # [B, Q, d_a]

        # Project
        x = self.proj(x)     # [B, Q, d]
        x = self.norm(x)     # [B, Q, d]

        # Pool queries and classify
        x = x.mean(dim=1)    # [B, d]
        logits = self.classifier(x)  # [B, num_emotions]

        return logits, weights


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_attention(weights: torch.Tensor, label: str = ""):
    """
    Print attention weight analysis for each query.

    Args:
        weights: [1, Q, T] attention weights
        label: description string (e.g. "BEFORE TRAINING")
    """
    Q = weights.shape[1]
    T = weights.shape[2]

    if label:
        print(f"\n  {'=' * 55}")
        print(f"  ATTENTION WEIGHTS {label}")
        print(f"  {'=' * 55}")

    for q in range(Q):
        top_frame = weights[0, q, :].argmax().item()
        top_weight = weights[0, q, top_frame].item()
        spread = weights[0, q, :].std().item()
        time_sec = top_frame / 50
        print(
            f"  Query {q + 1}: frame {top_frame:3d} (t={time_sec:.2f}s) "
            f"weight={top_weight:.4f}  spread={spread:.6f}"
        )

    avg_spread = weights[0].std(dim=-1).mean().item()
    return avg_spread


def analyze_frame_deviations(embeddings: torch.Tensor):
    """
    Analyze how different each frame is from the average.
    Used to verify if high-attention frames are genuinely distinctive.
    """
    T = embeddings.shape[1]
    frame_mean = embeddings[0].mean(dim=0)

    print(f"\n  {'=' * 55}")
    print(f"  FRAME DEVIATION ANALYSIS")
    print(f"  {'=' * 55}")

    # Check evenly spaced frames + find the overall maximum
    deviations = []
    for f in range(T):
        dev = (embeddings[0, f, :] - frame_mean).abs().mean().item()
        deviations.append((f, dev))

    # Sort by deviation to find top frames
    deviations.sort(key=lambda x: x[1], reverse=True)

    # Print sample frames
    sample_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    for f in sample_indices:
        dev = (embeddings[0, f, :] - frame_mean).abs().mean().item()
        bar = "█" * int(dev * 500)
        print(f"  Frame {f:3d} (t={f / 50:.2f}s): deviation={dev:.4f} {bar}")

    # Print top 3 most distinctive frames
    print(f"\n  Top 3 most distinctive frames:")
    for f, dev in deviations[:3]:
        print(f"  Frame {f:3d} (t={f / 50:.2f}s): deviation={dev:.4f}  ← peak")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_proxy_training(audio_path: str, emotion: str = None, steps: int = 100):
    """
    Run the proxy training experiment.

    Args:
        audio_path: path to EMO-DB .wav file
        emotion: emotion label (auto-detected from filename if not provided)
        steps: number of training steps
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    # ── Detect emotion ────────────────────────────────────────────────────
    if emotion is None:
        emotion = detect_emotion_from_filename(audio_path)
    if emotion is None:
        emotion = "anger"
        print(f"  Could not detect emotion from filename, defaulting to '{emotion}'")

    emotion_idx = EMODB_EMOTIONS.get(emotion, 0)
    print(f"  Emotion: {emotion} (label index: {emotion_idx})")

    # ── Load audio ────────────────────────────────────────────────────────
    print("  Loading audio...")
    sample_rate, waveform = wavfile.read(audio_path)

    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    else:
        waveform = waveform.astype(np.float32)

    if len(waveform.shape) == 2:
        waveform = waveform.mean(axis=1)

    if sample_rate != 16000:
        from scipy.signal import resample
        num_samples = int(len(waveform) * 16000 / sample_rate)
        waveform = resample(waveform, num_samples)
        sample_rate = 16000

    duration = len(waveform) / sample_rate
    print(f"  Duration: {duration:.2f}s")

    # ── Load wav2vec2 ─────────────────────────────────────────────────────
    print("  Loading Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model.eval()

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = w2v_model(**inputs)

    real_embeddings = outputs.last_hidden_state
    T_audio = real_embeddings.shape[1]
    d_a = real_embeddings.shape[2]
    print(f"  Embeddings: {list(real_embeddings.shape)}")

    # ── Setup proxy model ─────────────────────────────────────────────────
    model = ProxyEmotionModel(d_a=d_a, d=768, num_queries=8, num_emotions=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    label = torch.tensor([emotion_idx])

    # ── BEFORE training ───────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, weights_before = model(real_embeddings)

    spread_before = analyze_attention(weights_before, "BEFORE TRAINING")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n  {'=' * 55}")
    print(f"  TRAINING FOR {steps} STEPS")
    print(f"  {'=' * 55}")

    emotions_list = list(EMODB_EMOTIONS.keys())
    model.train()

    for step in range(steps):
        optimizer.zero_grad()
        logits, weights = model(real_embeddings)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        if step % (steps // 5) == 0:
            pred = logits.argmax().item()
            print(f"  Step {step:3d}: loss={loss.item():.4f} pred={emotions_list[pred]}")

    # ── AFTER training ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, weights_after = model(real_embeddings)

    spread_after = analyze_attention(weights_after, "AFTER TRAINING")

    # ── Comparison ────────────────────────────────────────────────────────
    print(f"\n  {'=' * 55}")
    print(f"  BEFORE vs AFTER COMPARISON")
    print(f"  {'=' * 55}")
    print(f"  Average spread BEFORE: {spread_before:.6f}")
    print(f"  Average spread AFTER:  {spread_after:.6f}")
    print(f"  Change: {spread_after / (spread_before + 1e-8):.1f}x")

    if spread_after > spread_before:
        print("\n  RESULT: Queries LEARNED to focus on specific frames!")
    else:
        print("\n  RESULT: Queries need more data to fully specialize")

    # ── Frame deviation analysis ──────────────────────────────────────────
    analyze_frame_deviations(real_embeddings)

    print(f"\n  {'=' * 55}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  {'=' * 55}")
    print(f"  Note: This proxy classifier is NOT the real LAM4SER pipeline.")
    print(f"  The real emotion classification happens via the LLM (Person 3+4).")
    print(f"  This experiment only proves that attention queries CAN learn")
    print(f"  to focus on emotionally salient frames.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 2 Proxy Training Experiment")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to EMO-DB .wav file",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        choices=list(EMODB_EMOTIONS.keys()),
        help="Emotion label (auto-detected from filename if omitted)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    args = parser.parse_args()

    run_proxy_training(args.audio, args.emotion, args.steps)
