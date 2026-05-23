"""
Person 2 — Compression Benchmark
=================================
Benchmarks all compression strategies on:
    1. Dummy tensors (no model download needed)
    2. Real wav2vec2 embeddings from EMO-DB audio files

Requirements:
    pip install torch scipy transformers

Usage:
    # Dummy benchmark only:
    python benchmark.py

    # Real wav2vec2 benchmark:
    python benchmark.py --audio path/to/emodb_file.wav

    # With German wav2vec2 model:
    python benchmark.py --audio path/to/file.wav --model german
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressors import (
    MeanPoolCompressor,
    ChunkPoolCompressor,
    AttentionPoolCompressor,
    ConvCompressor,
    AudioProjection,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_params(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def variance_retained(original: torch.Tensor, compressed: torch.Tensor) -> float:
    """
    Proxy for information retention.
    Compares variance of compressed output vs original.
    Higher ratio = more information preserved.
    """
    orig_var = original.var().item()
    comp_var = compressed.var().item()
    return comp_var / (orig_var + 1e-8)


def frame_deviation_analysis(embeddings: torch.Tensor, sample_frames: list = None):
    """
    Measure how acoustically different each frame is from the average.
    High deviation = unusual/emotionally distinctive moment.

    Args:
        embeddings: [1, T, d] audio embeddings
        sample_frames: list of frame indices to analyze (default: evenly spaced)
    """
    T = embeddings.shape[1]
    frame_mean = embeddings[0].mean(dim=0)  # average frame [d]

    if sample_frames is None:
        sample_frames = [0, T // 5, 2 * T // 5, 3 * T // 5, 4 * T // 5, T - 1]

    print(f"\n{'Frame':<10} {'Time':<10} {'Deviation':<12} Visualization")
    print("-" * 65)

    for f in sample_frames:
        if f >= T:
            continue
        frame = embeddings[0, f, :]
        dev = (frame - frame_mean).abs().mean().item()
        bar = "█" * int(dev * 500)
        print(f"Frame {f:<4} (t={f / 50:.2f}s)  {dev:<12.4f} {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 1: Dummy Tensors
# ─────────────────────────────────────────────────────────────────────────────

def run_dummy_benchmark():
    """Benchmark all strategies on simulated wav2vec2 output."""

    torch.manual_seed(42)

    B = 4
    T_audio = 75       # ~1.5 seconds at 50 fps
    d_a = 768          # wav2vec2-base hidden size
    d = 768            # LLM hidden size

    dummy = torch.randn(B, T_audio, d_a)

    print("=" * 72)
    print("  BENCHMARK 1 — DUMMY TENSORS")
    print("=" * 72)
    print(f"  Input shape: [{B}, {T_audio}, {d_a}]")
    print()

    strategies = [
        ("MeanPool (1 token)", MeanPoolCompressor()),
        ("ChunkPool chunk=5", ChunkPoolCompressor(chunk_size=5)),
        ("ChunkPool chunk=10", ChunkPoolCompressor(chunk_size=10)),
        ("ChunkPool chunk=15", ChunkPoolCompressor(chunk_size=15)),
        ("AttnPool Q=4", AttentionPoolCompressor(d=d_a, num_queries=4)),
        ("AttnPool Q=8", AttentionPoolCompressor(d=d_a, num_queries=8)),
        ("AttnPool Q=16", AttentionPoolCompressor(d=d_a, num_queries=16)),
        ("Conv stride=2", ConvCompressor(d=d_a, stride=2)),
        ("Conv stride=4", ConvCompressor(d=d_a, stride=4)),
        ("Conv stride=8", ConvCompressor(d=d_a, stride=8)),
    ]

    header = f"  {'Strategy':<24} {'Out Shape':<18} {'Tokens':<8} {'Ratio':<8} {'Params':>10}  {'VarRatio':<10} Status"
    print(header)
    print("  " + "-" * 90)

    results = []
    all_pass = True

    for name, compressor in strategies:
        try:
            proj = AudioProjection(d_a=d_a, d=d, compressor=compressor)
            proj.eval()
            with torch.no_grad():
                out = proj(dummy)

            tokens = out.shape[1]
            ratio = T_audio / tokens
            params = count_params(proj)
            var_r = variance_retained(dummy, out)
            shape_str = str(list(out.shape))
            ok = out.shape == (B, tokens, d)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False

            print(
                f"  {name:<24} {shape_str:<18} {tokens:<8} "
                f"{ratio:<8.1f} {params:>10,}  {var_r:<10.3f} {status}"
            )
            results.append((name, tokens, ratio, params, var_r))

        except Exception as e:
            print(f"  {name:<24} ERROR: {e}")
            all_pass = False

    print()
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ── Mismatched dimensions test ────────────────────────────────────────
    print()
    print("=" * 72)
    print("  RESEARCH 1: d_a != d (wav2vec2-large 1024 → GPT-2 768)")
    print("=" * 72)

    d_a_large = 1024
    dummy_large = torch.randn(B, T_audio, d_a_large)

    for name, compressor in [
        ("MeanPool", MeanPoolCompressor()),
        ("ChunkPool chunk=5", ChunkPoolCompressor(chunk_size=5)),
        ("AttnPool Q=8", AttentionPoolCompressor(d=d_a_large, num_queries=8)),
        ("Conv stride=4", ConvCompressor(d=d_a_large, stride=4)),
    ]:
        try:
            proj = AudioProjection(d_a=d_a_large, d=d, compressor=compressor)
            proj.eval()
            with torch.no_grad():
                out = proj(dummy_large)
            status = "PASS" if out.shape[-1] == d else "FAIL"
            print(
                f"  {name:<22} {list(dummy_large.shape)} -> {list(out.shape)}  {status}"
            )
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")

    # ── Information retention ─────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  RESEARCH 2: Information retention vs compression ratio")
    print("=" * 72)
    print(f"  {'Strategy':<24} {'Tokens':<8} {'Ratio':<8} {'VarRatio':<12} Interpretation")
    print("  " + "-" * 72)

    for name, tokens, ratio, params, var_r in sorted(results, key=lambda r: r[1], reverse=True):
        if var_r > 0.8:
            interp = "high info retained"
        elif var_r > 0.4:
            interp = "moderate retention"
        else:
            interp = "lossy"
        print(f"  {name:<24} {tokens:<8} {ratio:<8.1f} {var_r:<12.3f} {interp}")

    # ── Parameter efficiency ──────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  RESEARCH 3: Parameter efficiency")
    print("=" * 72)

    for name, tokens, ratio, params, var_r in sorted(results, key=lambda r: r[3]):
        bar = "#" * max(1, params // 200_000)
        print(f"  {name:<24} {params:>10,} params  {bar}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 2: Real Wav2Vec2 Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def run_real_benchmark(audio_path: str, model_name: str = "base"):
    """
    Benchmark compression on real wav2vec2 embeddings from an EMO-DB file.

    Args:
        audio_path: path to a .wav file (16kHz mono preferred)
        model_name: "base" for wav2vec2-base-960h,
                    "german" for wav2vec2-large-xlsr-53-german
    """
    import numpy as np
    from scipy.io import wavfile
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    # ── Model selection ───────────────────────────────────────────────────
    model_map = {
        "base": "facebook/wav2vec2-base-960h",
        "german": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    }
    model_id = model_map.get(model_name, model_name)

    # Expected hidden dimensions
    d_a_map = {
        "facebook/wav2vec2-base-960h": 768,
        "jonatasgrosman/wav2vec2-large-xlsr-53-german": 1024,
    }
    d_a = d_a_map.get(model_id, 768)
    d = 768  # LLM hidden size

    # ── Load audio ────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  BENCHMARK 2 — REAL WAV2VEC2 EMBEDDINGS")
    print("=" * 72)
    print(f"  Audio: {audio_path}")
    print(f"  Model: {model_id}")
    print()

    print("  Loading audio...")
    sample_rate, waveform = wavfile.read(audio_path)

    # Convert to float32 and normalize
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    else:
        waveform = waveform.astype(np.float32)

    # Stereo to mono
    if len(waveform.shape) == 2:
        waveform = waveform.mean(axis=1)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        from scipy.signal import resample

        num_samples = int(len(waveform) * 16000 / sample_rate)
        waveform = resample(waveform, num_samples)
        sample_rate = 16000

    duration = len(waveform) / sample_rate
    print(f"  Duration: {duration:.2f}s | Samples: {len(waveform)} | SR: {sample_rate}")

    # ── Load wav2vec2 ─────────────────────────────────────────────────────
    print(f"  Loading {model_id}...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id)
    model.eval()
    print("  Model loaded!")

    # ── Extract embeddings ────────────────────────────────────────────────
    print("  Extracting embeddings...")
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    real_embeddings = outputs.last_hidden_state
    T_audio = real_embeddings.shape[1]
    actual_d_a = real_embeddings.shape[2]

    print(f"  Embeddings: {list(real_embeddings.shape)}")
    print(f"  Tokens: {T_audio} (~{T_audio / duration:.0f} per second)")
    print()

    # ── Compression benchmark ─────────────────────────────────────────────
    strategies = [
        ("MeanPool", MeanPoolCompressor()),
        ("ChunkPool 5", ChunkPoolCompressor(chunk_size=5)),
        ("ChunkPool 10", ChunkPoolCompressor(chunk_size=10)),
        ("AttnPool Q=4", AttentionPoolCompressor(d=actual_d_a, num_queries=4)),
        ("AttnPool Q=8", AttentionPoolCompressor(d=actual_d_a, num_queries=8)),
        ("AttnPool Q=16", AttentionPoolCompressor(d=actual_d_a, num_queries=16)),
        ("Conv stride=2", ConvCompressor(d=actual_d_a, stride=2)),
        ("Conv stride=4", ConvCompressor(d=actual_d_a, stride=4)),
    ]

    print(f"  {'Strategy':<18} {'Input':<18} {'Output':<18} {'Ratio':<8} Status")
    print("  " + "-" * 68)

    for name, compressor in strategies:
        try:
            proj = AudioProjection(d_a=actual_d_a, d=d, compressor=compressor)
            proj.eval()
            with torch.no_grad():
                out = proj(real_embeddings)
            ratio = T_audio / out.shape[1]
            print(
                f"  {name:<18} {str(list(real_embeddings.shape)):<18} "
                f"{str(list(out.shape)):<18} {ratio:<8.1f} PASS"
            )
        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")

    # ── Attention weight analysis ─────────────────────────────────────────
    print()
    print("=" * 72)
    print("  ATTENTION WEIGHT ANALYSIS (AttnPool Q=8)")
    print("=" * 72)

    attn = AttentionPoolCompressor(d=actual_d_a, num_queries=8)

    with torch.no_grad():
        weights = attn.get_attention_weights(real_embeddings)  # [1, 8, T]

    print(f"  Audio duration: {duration:.2f}s | Total frames: {T_audio}")
    print()

    for q in range(8):
        top_frame = weights[0, q, :].argmax().item()
        top_weight = weights[0, q, top_frame].item()
        time_sec = top_frame / 50
        print(
            f"  Query {q + 1}: frame {top_frame:3d} (t={time_sec:.2f}s) "
            f"weight={top_weight:.4f}"
        )

    # ── Frame deviation analysis ──────────────────────────────────────────
    print()
    print("=" * 72)
    print("  FRAME DEVIATION ANALYSIS")
    print("  (Which frames are most acoustically distinctive?)")
    print("=" * 72)

    frame_deviation_analysis(
        real_embeddings,
        sample_frames=[0, T_audio // 5, 2 * T_audio // 5, 3 * T_audio // 5,
                       4 * T_audio // 5, T_audio - 1],
    )

    # ── Final output ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  FINAL OUTPUT TO PERSON 3")
    print("=" * 72)

    proj_final = AudioProjection(
        d_a=actual_d_a,
        d=d,
        compressor=AttentionPoolCompressor(d=actual_d_a, num_queries=8),
    )
    proj_final.eval()
    with torch.no_grad():
        final_out = proj_final(real_embeddings)

    print(f"  Input  (Person 1): {list(real_embeddings.shape)}")
    print(f"  Output (Person 3): {list(final_out.shape)}")
    print(f"  Compression:       {T_audio / final_out.shape[1]:.1f}x")
    print(f"  Value range:       [{final_out.min():.3f}, {final_out.max():.3f}]")
    print()
    print("  Done!")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 2 Compression Benchmark")
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to a .wav file for real wav2vec2 benchmark",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "german"],
        help="Wav2Vec2 model variant: base (English) or german",
    )
    args = parser.parse_args()

    # Always run dummy benchmark
    run_dummy_benchmark()

    # Run real benchmark if audio file provided
    if args.audio:
        run_real_benchmark(args.audio, args.model)
    else:
        print()
        print("  Tip: run with --audio <path> for real wav2vec2 benchmark")
        print("  Example: python benchmark.py --audio data/emodb/03a05Wb.wav")
