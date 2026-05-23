# Compression & Projection (Person 2)

## Overview

This module compresses audio embeddings from wav2vec2 before passing them to the LLM via cross-attention (Person 3). It sits between Person 1 (audio encoding) and Person 3 (audio-LLM fusion) in the LAM4SER pipeline.

**Pipeline position:**
```
Person 1 (wav2vec2) → Person 2 (compression) → Person 3 (cross-attention)
```

## Problem

wav2vec2 outputs ~50 frames per second. A 3-second audio clip produces 147 tokens, each a vector of 768 (or 1024) numbers. Feeding all of these into the LLM is slow and memory-heavy because cross-attention cost scales as O(T_text × T_audio). Most of these frames don't carry emotional information anyway — research shows only 15% of frames contain 80% of emotional cues (Leygue et al., Interspeech 2025).

Our job: compress 147 tokens into something manageable without losing the emotional peaks.

## Strategies Implemented

| Strategy | Output Tokens | Compression | Learnable | Params (extra) |
|---|---|---|---|---|
| MeanPool | 1 | 147x | No | 0 |
| ChunkPool (k=5) | 29 | 5x | No | 0 |
| ChunkPool (k=10) | 14 | 10x | No | 0 |
| AttentionPool (Q=8) | 8 | 18.4x | Yes | 6,144 |
| Conv stride=4 | 37 | 4x | Yes | 4,131,072 |

## Recommended Default

**AttentionPool Q=8** — best tradeoff of compression ratio, parameter efficiency, and ability to learn emotionally salient frames.

## Key Findings

1. **All strategies pass** on both dummy tensors and real wav2vec2 embeddings from EMO-DB
2. **Attention pooling adds only 6,144 extra parameters** vs mean pooling — but is learnable
3. **Conv strategies are expensive** — up to 9.4M parameters for stride=8
4. **Proxy training shows queries converge** to acoustically distinctive frames (frame 89 at t=1.78s for anger)
5. **Literature support**: Leygue et al. (Interspeech 2025) found 15% of frames carry 80% of emotional information, validating our attention pooling approach

## Output Contract to Person 3

```
Input:  [B, T_audio, d_a]   e.g. [4, 147, 768]
Output: [B, T_compressed, d] e.g. [4, 8, 768]
```

The output tensor is used as K and V in Person 3's cross-attention module.

## Usage

```python
from models.compression import AudioProjection, AttentionPoolCompressor

# Default setup (wav2vec2-base → same-dim LLM)
compressor = AttentionPoolCompressor(d=768, num_queries=8)
proj = AudioProjection(d_a=768, d=768, compressor=compressor)

audio_embeddings = ...  # [B, T_audio, 768] from Person 1
compressed = proj(audio_embeddings)  # [B, 8, 768] → to Person 3

# Mismatched dimensions (wav2vec2-large → smaller LLM)
compressor = AttentionPoolCompressor(d=1024, num_queries=8)
proj = AudioProjection(d_a=1024, d=768, compressor=compressor)
```

## Running Benchmarks

```bash
# Dummy tensor benchmark (no model download)
python benchmark.py

# Real wav2vec2 on EMO-DB file
python benchmark.py --audio ../../data/emodb/03a05Wb.wav

# With German wav2vec2 model
python benchmark.py --audio ../../data/emodb/03a05Wb.wav --model german
```

## Running Proxy Training

```bash
# Auto-detects emotion from filename (W = Wut = anger)
python proxy_training.py --audio ../../data/emodb/03a05Wb.wav

# Explicit emotion + more steps
python proxy_training.py --audio ../../data/emodb/03a05Wb.wav --emotion anger --steps 200
```

## File Structure

```
models/compression/
├── __init__.py            # Module exports
├── compressors.py         # 4 strategies + AudioProjection
├── benchmark.py           # Dummy + real wav2vec2 benchmarks
├── proxy_training.py      # Proxy training experiment
└── README.md              # This file
```

## Sprint 2 Plans

- [ ] Switch to German wav2vec2 (`jonatasgrosman/wav2vec2-large-xlsr-53-german`)
- [ ] Benchmark across all 1,632 EMO-DB files
- [ ] Compare attention patterns across different emotions
- [ ] Integrate with full pipeline (Person 3 cross-attention + Person 4 training)
- [ ] Measure impact of compression on final emotion classification F1

## References

- Leygue et al. (2025). "Explainable Speech Emotion Recognition Through Attentive Pooling." Interspeech 2025.
- Casals-Salvador et al. (2024). "BSC-UPC at EmoSPeech-IberLEF2024: Attention Pooling for Emotion Recognition."
- Costa et al. (2024). "Double Multi-Head Attention Multimodal System for Odyssey 2024 SER Challenge."
