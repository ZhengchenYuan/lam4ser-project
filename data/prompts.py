"""
Prompt templates for LAM4SER.

This file centralizes all prompt variants used by both:
1. classifier-based AudioGPT2
2. autoregressive label generation
"""

# EMoDB (7-class):
# LABELS = [
#     "anger",
#     "boredom",
#     "disgust",
#     "fear",
#     "happiness",
#     "neutral",
#     "sadness",
# ]

# AIBO (5-class, IS2009 Emotion Challenge):
LABELS = [
    "anger",
    "emphatic",
    "neutral",
    "positive",
    "rest",
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
            - generation
            - feature_generation

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
