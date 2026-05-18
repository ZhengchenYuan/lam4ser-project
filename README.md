# lam4ser-project

Large Audio Models for Speech Emotion Recognition.

This repository contains the group implementation for the ASL project.

## Current focus

- preprocessing and audio embedding extraction
- audio token compression and projection
- audio-LLM fusion with cross-attention/adapters
- training and evaluation on SER datasets

## Initial dataset

- EMODB

## Modules

- `data/`: dataset loading and preprocessing
- `models/audio_encoder/`: wav2vec2 / HuBERT feature extraction
- `models/compression/`: pooling, projection, learned compression
- `models/fusion/`: cross-attention and adapter modules
- `training/`: training loop and optimization
- `evaluation/`: metrics and visualization
- `docs/`: notes, weekly reports, slides
