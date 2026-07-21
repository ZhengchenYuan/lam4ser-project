# AIBO label-blind acoustic-rationale SFT

This branch tests whether label-blind acoustic-rationale supervision improves AIBO UAR
over the existing controls:

- no-cue answer-only + mixed-effects: UAR 0.4108
- balanced mixed template reasoning: UAR 0.3847

The new experiment keeps audio fusion enabled and uses a fixed text prompt
without handcrafted cue text. Its target is:

```text
<think>label-blind reasoning grounded in available prosody</think>
<answer>label</answer>
```

## Scope

The annotation generator uses only evidence currently available in this repo:
speaker identity, numeric acoustic features, and mixed-effects
speaker-relative prosody. It uses deterministic local rules and makes no API
calls. Transcript, age, gender, word stress, and intonation contour are not
used and are recorded as unavailable.
Feature directions are normalized with mixed-effects residual standard
deviations estimated from training-speaker neutral utterances.
This is structured acoustic-rationale SFT, not an EmotionThinker/GRPO-PTR reproduction.

## Commands

Generate the frozen, resumable JSONL locally with no API key or API cost:

```bash
sbatch scripts/generate_aibo_practical_cot.sbatch
```

Output: `annotations/aibo_label_blind_rationale_v1.jsonl`.

Review the annotation file before training. Then run:

```bash
sbatch scripts/train_aibo_full_cot_balanced_mixed_effects.sbatch
sbatch scripts/eval_aibo_full_cot_balanced_mixed_effects.sbatch
```

Primary metric: UAR. Accuracy, weighted F1, format validity, generated
reasoning, and per-class metrics remain diagnostic outputs.

Expected checkpoint:

```text
checkpoints_AIBO/wavlm-large_speaker_label_blind_rationale_generation_neutral_mixed_effects_weighted_balanced_p0.5_m2.0_generation_best.pt
```
