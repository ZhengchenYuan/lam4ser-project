"""
Person 2 — Compression Strategies & Audio Projection
=====================================================
This module implements four audio token compression strategies
and a unified AudioProjection module that combines compression
with linear projection for LLM integration.

Problem:
    wav2vec2 outputs ~50 frames/second → a 3s clip = 147 tokens.
    Cross-attention cost = O(T_text × T_audio).
    We need to reduce T_audio without losing emotional information.

Recommendation:
    AttentionPoolCompressor with Q=8 queries gives the best balance
    of compression (18.4x), parameter efficiency (+6,144 params),
    and ability to learn emotionally salient frames.

    Supported by Leygue et al. (Interspeech 2025): only 15% of frames
    carry 80% of emotional information → attention pooling captures this.

Output contract to Person 3 (cross-attention):
    Input:  [B, T_audio, d_a]   e.g. [4, 147, 768]
    Output: [B, T_compressed, d] e.g. [4, 8, 768]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: Mean Pooling
# ─────────────────────────────────────────────────────────────────────────────

class MeanPoolCompressor(nn.Module):
    """
    Collapses ALL audio frames into a single vector via mean pooling.

    Compression: T_audio → 1 token
    Parameters:  0 (no learnable params)

    Pros:
        - Simplest possible baseline
        - Very fast, zero overhead
    Cons:
        - Loses all temporal structure
        - One token may be too aggressive for emotion recognition
        - Cannot distinguish beginning/middle/end of utterance

    Example:
        >>> comp = MeanPoolCompressor()
        >>> x = torch.randn(4, 147, 768)    # 4 samples, 147 tokens
        >>> out = comp(x)
        >>> out.shape                         # [4, 1, 768]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d] audio embeddings
        Returns:
            [B, 1, d] single mean-pooled vector
        """
        return x.mean(dim=1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: Chunk Pooling
# ─────────────────────────────────────────────────────────────────────────────

class ChunkPoolCompressor(nn.Module):
    """
    Divides frames into fixed-size chunks and mean-pools each chunk.

    Compression: T_audio → T_audio // chunk_size tokens
    Parameters:  0 (no learnable params)

    Pros:
        - Preserves coarse temporal structure (beginning/middle/end)
        - Simple, predictable output size
        - Adjustable compression via chunk_size
    Cons:
        - Treats all frames within a chunk equally
        - Cannot learn which frames are emotionally important
        - Tail frames may be dropped if T not divisible by chunk_size

    Example:
        >>> comp = ChunkPoolCompressor(chunk_size=5)
        >>> x = torch.randn(4, 147, 768)
        >>> out = comp(x)
        >>> out.shape                         # [4, 29, 768]  (147//5 = 29)
    """

    def __init__(self, chunk_size: int = 5):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d] audio embeddings
        Returns:
            [B, T // chunk_size, d] chunk-pooled embeddings
        """
        B, T, d = x.shape

        # Trim to nearest multiple of chunk_size
        # e.g. T=147, chunk=5 → T_trim=145 (drop last 2 frames)
        T_trim = (T // self.chunk_size) * self.chunk_size
        x = x[:, :T_trim, :]

        # Reshape into chunks: [B, num_chunks, chunk_size, d]
        x = x.view(B, T_trim // self.chunk_size, self.chunk_size, d)

        # Average each chunk: [B, num_chunks, d]
        return x.mean(dim=2)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3: Attention Pooling (Recommended)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPoolCompressor(nn.Module):
    """
    Learns which audio frames matter most via dot-product attention.
    A fixed set of learnable query vectors attends over all audio frames.

    Compression: T_audio → num_queries tokens
    Parameters:  num_queries × d (e.g. 8 × 768 = 6,144 params)

    Each query can specialise on different aspects:
        - Query 1 might focus on pitch peaks
        - Query 2 might focus on energy bursts
        - Query 3 might focus on silence boundaries
        - Query 4 might focus on stressed syllables

    Supported by literature:
        - Leygue et al. (Interspeech 2025): 15% of frames carry 80%
          of emotional information → attention can capture this
        - Casals-Salvador et al. (IberLEF 2024): attention pooling
          achieved 86.69% F1, first place in multimodal SER task

    Pros:
        - Learnable: adapts to emotionally salient frames
        - Tiny parameter overhead (6,144 for Q=8)
        - Strong compression (147→8 = 18.4x)
        - Biologically plausible (mirrors human listening strategies)
    Cons:
        - Requires training to specialize (random before training)
        - May suffer query collapse on small datasets (all queries
          converging to same frames)

    Example:
        >>> comp = AttentionPoolCompressor(d=768, num_queries=8)
        >>> x = torch.randn(4, 147, 768)
        >>> out = comp(x)
        >>> out.shape                         # [4, 8, 768]
    """

    def __init__(self, d: int, num_queries: int = 8):
        super().__init__()
        # Learnable query vectors, initialized small for stable training
        self.queries = nn.Parameter(torch.randn(num_queries, d) * 0.02)
        # Scale factor to prevent attention scores from exploding
        # Standard from "Attention Is All You Need" (Vaswani et al., 2017)
        self.scale = d ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d] audio embeddings
        Returns:
            [B, num_queries, d] attention-pooled embeddings
        """
        # Compute attention scores: how much each query attends to each frame
        # queries: [Q, d], x: [B, T, d] → scores: [B, Q, T]
        scores = torch.einsum('qd,btd->bqt', self.queries, x) * self.scale

        # Softmax over T dimension → attention weights sum to 1 per query
        weights = F.softmax(scores, dim=-1)  # [B, Q, T]

        # Weighted sum of frames for each query
        # weights: [B, Q, T], x: [B, T, d] → out: [B, Q, d]
        out = torch.einsum('bqt,btd->bqd', weights, x)

        return out

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw attention weights for visualization/analysis.

        Args:
            x: [B, T, d] audio embeddings
        Returns:
            weights: [B, num_queries, T] attention weights
        """
        scores = torch.einsum('qd,btd->bqt', self.queries, x) * self.scale
        return F.softmax(scores, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4: 1D Convolutional Downsampling
# ─────────────────────────────────────────────────────────────────────────────

class ConvCompressor(nn.Module):
    """
    Strided 1D convolution: reduces T by a fixed stride factor.
    Uses overlapping windows (kernel_size = 2*stride - 1).

    Compression: T_audio → ~T_audio // stride tokens
    Parameters:  kernel_size × d × d (can be very large!)

    Pros:
        - Preserves local temporal patterns
        - Learnable: adapts asymmetric weights within each window
        - Good for capturing how speech changes frame-to-frame
    Cons:
        - Very parameter-heavy (e.g. stride=4 → 4.7M params)
        - Less compression than attention pooling
        - More expensive to train

    Example:
        >>> comp = ConvCompressor(d=768, stride=4)
        >>> x = torch.randn(4, 147, 768)
        >>> out = comp(x)
        >>> out.shape                         # [4, 37, 768]
    """

    def __init__(self, d: int, stride: int = 4):
        super().__init__()
        kernel = stride * 2 - 1
        self.conv = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d] audio embeddings
        Returns:
            [B, T // stride, d] conv-downsampled embeddings
        """
        # Conv1d expects [B, channels, T], not [B, T, channels]
        x = x.transpose(1, 2)   # [B, d, T]
        x = self.conv(x)        # [B, d, T // stride]
        x = x.transpose(1, 2)   # [B, T // stride, d]
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PROJECTION (Full Person 2 Module)
# ─────────────────────────────────────────────────────────────────────────────

class AudioProjection(nn.Module):
    """
    Full Person 2 pipeline: compression + projection + normalization.

    Pipeline:
        [B, T_audio, d_a]
            → compressor  → [B, T_compressed, d_a]  (reduce tokens)
            → linear proj → [B, T_compressed, d]     (match LLM dim)
            → LayerNorm   → [B, T_compressed, d]     (normalize)

    The output tensor becomes K and V in Person 3's cross-attention.

    Handles mismatched dimensions:
        wav2vec2-base:  d_a=768  → d=768  (same)
        wav2vec2-large: d_a=1024 → d=768  (projection needed)

    Example:
        >>> compressor = AttentionPoolCompressor(d=768, num_queries=8)
        >>> proj = AudioProjection(d_a=768, d=768, compressor=compressor)
        >>> audio = torch.randn(4, 147, 768)
        >>> out = proj(audio)
        >>> out.shape                         # [4, 8, 768]
    """

    def __init__(self, d_a: int, d: int, compressor: nn.Module):
        """
        Args:
            d_a: input dimension (wav2vec2 hidden size, e.g. 768 or 1024)
            d:   output dimension (LLM hidden size, e.g. 768)
            compressor: any compression strategy from above
        """
        super().__init__()
        self.compressor = compressor
        self.proj = nn.Linear(d_a, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [B, T_audio, d_a] from Person 1
        Returns:
            compressed:     [B, T_compressed, d] ready for Person 3
        """
        x = self.compressor(audio_features)  # [B, T_compressed, d_a]
        x = self.proj(x)                     # [B, T_compressed, d]
        x = self.norm(x)                     # [B, T_compressed, d]
        return x
