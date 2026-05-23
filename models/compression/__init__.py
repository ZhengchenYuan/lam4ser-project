"""
Person 2 — Compression & Projection Module
============================================
Reduces audio token count from wav2vec2 output and projects
into LLM-compatible dimensions.

Usage:
    from models.compression import AudioProjection, AttentionPoolCompressor

    compressor = AttentionPoolCompressor(d=768, num_queries=8)
    proj = AudioProjection(d_a=768, d=768, compressor=compressor)
    compressed = proj(audio_embeddings)  # [B, T_audio, 768] -> [B, 8, 768]
"""

from .compressors import (
    MeanPoolCompressor,
    ChunkPoolCompressor,
    AttentionPoolCompressor,
    ConvCompressor,
    AudioProjection,
)

__all__ = [
    "MeanPoolCompressor",
    "ChunkPoolCompressor",
    "AttentionPoolCompressor",
    "ConvCompressor",
    "AudioProjection",
]
