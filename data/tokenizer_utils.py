from transformers import GPT2Tokenizer


STRUCTURED_SPECIAL_TOKENS = [
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<caption>",
    "</caption>",
    "<evidence>",
    "</evidence>",
]

CUE_SPECIAL_TOKENS = [
    "<pitch>",
    "</pitch>",
    "<energy>",
    "</energy>",
    "<rhythm>",
    "</rhythm>",
    "<duration>",
    "</duration>",
]


TOKENIZER_DIAGNOSTIC_TEXTS = [
    "<answer>anger</answer>",
    "<think>The speech has high energy.</think><answer>anger</answer>",
    "<caption>high pitch, high energy</caption><answer>anger</answer>",
    "<answer>anger</answer><evidence>High energy suggests strong activation.</evidence>",
    "<caption><pitch>higher</pitch><energy>lower</energy><rhythm>faster</rhythm><duration>similar</duration></caption>",
]


def build_generation_tokenizer(
    add_structured_tokens: bool = True,
    include_cue_tokens: bool = False,
    verbose: bool = False,
):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    before_len = len(tokenizer)
    added_count = 0

    if add_structured_tokens:
        special_tokens = list(STRUCTURED_SPECIAL_TOKENS)
        if include_cue_tokens:
            special_tokens.extend(CUE_SPECIAL_TOKENS)

        added_count = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
    else:
        special_tokens = []

    after_len = len(tokenizer)

    if verbose:
        print("Generation tokenizer configuration:")
        print(f"  Structured special tokens: {special_tokens}")
        print(f"  Vocab size before:        {before_len}")
        print(f"  Added tokens:             {added_count}")
        print(f"  Vocab size after:         {after_len}")

    return tokenizer


def print_tokenizer_diagnostics():
    print("GPT-2 tokenizer diagnostics before adding structured special tokens:")
    base_tokenizer = build_generation_tokenizer(
        add_structured_tokens=False,
        verbose=True,
    )
    _print_examples(base_tokenizer)

    print()
    print("GPT-2 tokenizer diagnostics after adding structured special tokens:")
    structured_tokenizer = build_generation_tokenizer(
        include_cue_tokens=True,
        add_structured_tokens=True,
        verbose=True,
    )
    _print_examples(structured_tokenizer)


def _print_examples(tokenizer):
    for text in TOKENIZER_DIAGNOSTIC_TEXTS:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print("-" * 80)
        print(f"Text:   {text}")
        print(f"IDs:    {token_ids}")
        print(f"Tokens: {tokens}")
