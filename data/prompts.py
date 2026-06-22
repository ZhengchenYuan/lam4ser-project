"""
Prompt templates for LAM4SER.

This file centralizes all prompt variants used by both:
1. classifier-based AudioGPT2
2. autoregressive label generation
"""

LABELS = [
    "anger",
    "boredom",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
]


LABEL_TEXT = ", ".join(LABELS)


PROMPTS = {
    "base": (
        "Classify the emotion of this speech:"
    ),

    "label_list": (
        "Classify the emotion of this speech. "
        f"Possible labels are {LABEL_TEXT}."
    ),

    "feature": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        f"Possible labels are {LABEL_TEXT}."
    ),

    "feature_speaker": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        f"Possible labels are {LABEL_TEXT}."
    ),

    "generation": (
        "Classify the emotion of this speech. "
        f"Possible labels are {LABEL_TEXT}. "
        "Answer with one label only:"
    ),

    "feature_generation": (
        "Classify the emotion of this speech. "
        "Acoustic features: {features}. "
        f"Possible labels are {LABEL_TEXT}. "
        "Answer with one label only:"
    ),

    "answer_generation": (
        "Describe the emotional speech and predict the emotion."
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


def get_prompt(prompt_type: str, features: str | None = None) -> str:
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
            - reasoning_generation_global
            - speaker_reasoning_generation
            - speaker_reasoning_generation_answer_first

        features:
            Textual acoustic feature description, for example:
            "high pitch, high energy, short duration"

    Returns:
        Prompt string.
    """
    if prompt_type not in PROMPTS:
        raise ValueError(
            f"Unknown prompt_type: {prompt_type}. "
            f"Available prompt types: {list(PROMPTS.keys())}"
        )

    template = PROMPTS[prompt_type]

    if "{features}" in template:
        if features is None:
            raise ValueError(
                f"Prompt type '{prompt_type}' requires acoustic feature text, "
                "but features=None was provided."
            )
        return template.format(features=features)

    return template
