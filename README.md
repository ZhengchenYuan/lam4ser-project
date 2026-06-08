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


---

## Yuan's Extensions

This part extends the WavLM-large AudioGPT2 setup with prompt variants, acoustic feature text, and autoregressive label generation.

### `data/`

`prompts.py`: stores all prompt templates used in the prompt ablation and generation experiments.

Supported prompt types:

- `base`: simple classifier prompt
- `label_list`: classifier prompt with all possible emotion labels
- `feature`: classifier prompt with acoustic feature text
- `generation`: autoregressive prompt for label generation
- `feature_generation`: autoregressive prompt with acoustic feature text

The acoustic feature text is inserted into the GPT-2 prompt, not into the WavLM audio embedding stream.

`generation_dataset.py`: dataset for autoregressive label generation. It builds `prompt + target label text` and masks the prompt tokens with `-100`, so the language-model loss is only computed on the target label tokens.

### `features/`

`acoustic_features.py`: extracts acoustic features from the original wav files using `librosa`.

Extracted features:

- duration: `librosa.get_duration`
- pitch mean/std: `librosa.pyin`
- energy mean/std: `librosa.feature.rms`
- tempo: `librosa.beat.tempo`

Each wav file is loaded as a 16 kHz mono waveform:

```python
librosa.load(str(wav_path), sr=16000, mono=True)
```

`feature_prompt.py`: converts numeric acoustic features into short text descriptions, for example:

```text
high pitch, moderate pitch variation, high energy, short duration
```

The acoustic feature cache is saved to:

```text
embeddings/wavlm-large_acoustic_features.pt
```

### `models/`

`audio_gpt2_generation.py`: autoregressive version of AudioGPT2.

The original classifier model predicts emotion classes with a classifier head:

```text
last token hidden state -> classifier head -> 7 emotion logits
```

The generation model predicts the emotion label as text:

```text
hidden states -> GPT-2 LM head -> generated label text
```

The target label space is still the same 7 EMoDB emotions, but the model outputs text such as `anger`, `neutral`, or `sadness`.

### `training/`

`train_base_model.py`: now supports prompt variants through `--prompt_type`.

Classifier prompt experiments:

```bash
python training/train_base_model.py --encoder wavlm-large --prompt_type base
python training/train_base_model.py --encoder wavlm-large --prompt_type label_list
python training/train_base_model.py --encoder wavlm-large --prompt_type feature
```

`train_generation_model.py`: trains the autoregressive label generation model.

Generation experiments:

```bash
python training/train_generation_model.py --encoder wavlm-large --prompt_type generation
python training/train_generation_model.py --encoder wavlm-large --prompt_type feature_generation
```

### `evaluation/`

`evaluate_generation.py`: evaluates generated emotion labels by mapping generated text back to the 7 emotion classes. It reports accuracy, weighted F1, generated label validity, prediction distribution, and confusion matrix.

Run with:

```bash
python evaluation/evaluate_generation.py --encoder wavlm-large --prompt_type generation
python evaluation/evaluate_generation.py --encoder wavlm-large --prompt_type feature_generation
```

---

## Yuan's Results

All experiments below use **WavLM-large embeddings** and the same speaker-independent EMoDB split.

```text
                                      Accuracy   Weighted F1   Validity
  classifier + base                    88.27%        87.99%          -
  classifier + label_list              87.04%        87.28%          -
  classifier + feature                 83.33%        83.18%          -
  generation                           89.51%        89.29%     100.00%
  feature_generation                   78.40%        78.32%     100.00%
```

Best result in this part: **autoregressive generation with WavLM-large, 89.51% accuracy / 89.29% weighted F1**.

Main observations:

- Autoregressive generation performs best within the WavLM-large setting.
- Adding the label list does not improve the classifier baseline.
- Acoustic feature text prompts reduce performance in both classifier and generation settings.
- Generated labels are valid, but greedy decoding often repeats the label.
