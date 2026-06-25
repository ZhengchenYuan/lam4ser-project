"""
Prompt templates for LAM4SER.

This file centralizes all prompt variants used by both:
1. classifier-based AudioGPT2
2. autoregressive label generation
"""

EMODB_LABELS = [
    "anger",
    "boredom",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
]

AIBO_LABELS = [
    "anger",
    "emphatic",
    "neutral",
    "positive",
    "rest",
]


LABELS = EMODB_LABELS
LABEL_TEXT = ", ".join(LABELS)


PROMPTS = {
    "base": (
        "Classify the emotion of this speech:"
    ),

    "label_list": (
        "Classify the emotion of this speech. "
        "Possible labels are {label_text}."
    ),

    "feature": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        "Possible labels are {label_text}."
    ),

    "feature_speaker": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        "Possible labels are {label_text}."
    ),

    "generation": (
        "Classify the emotion of this speech. "
        "Possible labels are {label_text}. "
        "Answer with one label only:"
    ),

    "feature_generation": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        "Possible labels are {label_text}. "
        "Answer with one label only:"
    ),

    "answer_generation": (
        "Describe the emotional speech and predict the emotion."
    ),

    "speaker_feature_answer_generation": (
        "Describe the emotional speech and predict the emotion. "
        "Speaker-relative acoustic cues: {features}."
    ),

    "speaker_feature_answer_caption_generation": (
        "Describe the emotional speech and predict the emotion. "
        "Speaker-relative acoustic cues: {features}."
    ),

    "speaker_feature_answer_evidence_generation": (
        "Describe the emotional speech and predict the emotion. "
        "Speaker-relative acoustic cues: {features}."
    ),

    "reasoning_generation_global": (
        "Describe the emotional speech and predict the emotion."
    ),

    "speaker_reasoning_generation": (
        "Describe the emotional speech and predict the emotion."
    ),

    "speaker_reasoning_generation_answer_first": (
        "Describe the emotional speech and predict the emotion."
    ),
}


def get_prompt(
    prompt_type: str,
    features: str | None = None,
    labels: list[str] | None = None,
) -> str:
    """
    Build a prompt string from the selected prompt type.

    Args:
        prompt_type:
            One of:
            - base
            - label_list
            - feature
            - feature_speaker
            - generation
            - feature_generation
            - answer_generation
            - speaker_feature_answer_generation
            - speaker_feature_answer_caption_generation
            - speaker_feature_answer_evidence_generation
            - reasoning_generation_global
            - speaker_reasoning_generation
            - speaker_reasoning_generation_answer_first

        features:
            Textual acoustic feature description, for example:
            "high pitch, high energy, short duration"

        labels:
            Optional dataset-specific labels to list in prompts. Defaults to
            EMoDB labels for backwards compatibility.

    Returns:
        Prompt string.
    """
    if prompt_type not in PROMPTS:
        raise ValueError(
            f"Unknown prompt_type: {prompt_type}. "
            f"Available prompt types: {list(PROMPTS.keys())}"
        )

    template = PROMPTS[prompt_type]

    label_text = ", ".join(labels if labels is not None else LABELS)
    text = template.replace("{label_text}", label_text)

    if "{features}" in text:
        if features is None:
            raise ValueError(
                f"Prompt type '{prompt_type}' requires acoustic feature text, "
                "but features=None was provided."
            )
        text = text.format(features=features)

    return text
