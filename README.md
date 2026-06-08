# lam4ser-project

Large Audio Models for Speech Emotion Recognition.

This repository contains the group implementation for the ASL project.

## Current focus

- preprocessing and audio embedding extraction
- audio token compression and projection
- audio-LLM fusion with cross-attention/adapters
- training and evaluation on SER datasets

---

## Dataset

**EMoDB** -- German emotional speech corpus, 816 samples, 7 emotion classes (anger, boredom, disgust, fear, happiness, neutral, sadness).

We use a speaker-independent split so the model never sees a speaker during training that it will be tested on.

- Train: speakers 11, 12, 13, 14, 15, 16 (493 samples)
- Val:   speakers 09, 10 (161 samples)
- Test:  speakers 03, 08 (162 samples)

Speaker IDs are extracted from the first two characters of the filename, which is how EMoDB encodes them.

---

## Modules

### `data/`

`dataset.py` handles loading the pre-extracted embeddings from disk and the speaker-independent split logic. The dataset returns three things per sample: the fixed text prompt as input_ids, the audio embedding, and the label.

### `models/audio_encoder/`

`preprocessing.py` runs offline (not during training). It loads the raw EMoDB audio files, passes each one through a chosen encoder, and saves the last hidden states to disk as a .pt file. The encoder is selected via `--encoder`:

```
python models/audio_encoder/preprocessing.py --encoder wavlm-large
python models/audio_encoder/preprocessing.py --encoder wav2vec2-large-emotion
python models/audio_encoder/preprocessing.py --encoder wav2vec2-base   # default
python models/audio_encoder/preprocessing.py --encoder hubert-large
```

Output goes to `embeddings/{encoder-name}_embeddings.pt`. Supported encoders and their output dimensions:

| Key | Model | Dim |
|---|---|---|
| `wav2vec2-base` | facebook/wav2vec2-base-960h | 768 |
| `wav2vec2-large-emotion` | audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | 1024 |
| `wavlm-large` | microsoft/wavlm-large | 1024 |
| `hubert-large` | facebook/hubert-large-ls960-ft | 1024 |


### `models/compression/`

`compressor.py` collapses variable-length audio sequences to a fixed length of 50 tokens using temporal mean pooling.

### `models/fusion/`

We inject audio information into GPT-2 at multiple layers using cross-attention adapters.

`cross_attention.py`: CrossAttentionAdapter takes text hidden states as query and audio hidden states as key/value, runs multi-head cross-attention, passes the output through a small bottleneck MLP, and adds it back to the text hidden states with a residual connection + LayerNorm.

`fusion_block.py`: thin wrapper around CrossAttentionAdapter, handles the case where audio_dim != text_dim with a linear projection.

### `models/`

`audio_gpt2.py`: the full model. GPT-2 is loaded and fully frozen (124M params, none trained). We attach one CrossAttentionAdapter after every third GPT-2 transformer block (layers 2, 5, 8, 11). After the final layer, the last token's hidden state goes through a small classifier head (LayerNorm -> Linear -> GELU -> Dropout -> Linear).

Optional LoRA support: pass `--lora_rank` to inject low-rank updates into the GPT-2 attention weights alongside the cross-attention adapters.

Trainable parameters: ~10M with 768-dim encoder, ~13.2M with 1024-dim encoder.

### `training/`

`train_base_model.py`: training loop. Pass `--encoder` to select which embeddings to train on; pass `--lora_rank` to enable LoRA. Key choices: AdamW lr=1e-5, linear warmup + decay, class-weighted cross-entropy, grad clipping at 1.0, batch size 8, 100 epochs. See `training_notes.txt` for the full run history.

### `evaluation/`

`evaluate.py`: computes accuracy, weighted F1, and confusion matrix on the test set using the saved best checkpoint.

`compare_encoders.py`: evaluates all trained checkpoints in one pass and produces a grouped bar chart and confusion matrix for the best encoder.

### `baselines/`

Three baselines for comparison, all using the same speaker-independent split:

`svm_mfcc.py`: SVM with 84 hand-crafted features (40 MFCCs mean+std, pitch, energy). Encoder-agnostic.

`embedding_probes.py`: linear probe and MLP probe trained on top of frozen encoder embeddings. Run for all four encoders.

`compare.py`: aggregates all baseline and AudioGPT2 results into a single comparison table and bar chart.

---

## Results

Full comparison on the test set (162 samples, speaker-independent split):

```
                        hubert-large   wav2vec2-base   wav2vec2-large-emotion   wavlm-large
  SVM + MFCC                  63.6%           63.6%                    63.6%         63.6%
  linear probe                75.3%           49.4%                    85.2%         87.0%
  MLP probe                   81.5%           42.0%                    85.8%         92.6%
  AudioGPT2                   78.4%           42.6%                    93.2%         88.3%
```

Best result: **AudioGPT2 + wav2vec2-large-emotion, 93.2% accuracy / 88.2% weighted F1**. This is also the one case where AudioGPT2 clearly beats the MLP probe. For WavLM-large and HuBERT-large the MLP probe wins, likely because 493 training samples is not enough for the cross-attention adapters to outperform a smaller model. Numbers will be lower on naturalistic datasets since EMoDB is acted speech in a clean studio. See `training_notes.txt` for the full run-by-run history.
