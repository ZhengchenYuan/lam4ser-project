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


TOKENIZER_DIAGNOSTIC_TEXTS = [
    "<answer>anger</answer>",
    "<think>The speech has high energy.</think><answer>anger</answer>",
    "<caption>high pitch, high energy</caption><answer>anger</answer>",
    "<answer>anger</answer><evidence>High energy suggests strong activation.</evidence>",
]


def build_generation_tokenizer(
    add_structured_tokens: bool = True,
    verbose: bool = False,
):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    before_len = len(tokenizer)
    added_count = 0

    if add_structured_tokens:
        added_count = tokenizer.add_special_tokens(
            {"additional_special_tokens": STRUCTURED_SPECIAL_TOKENS}
        )

    after_len = len(tokenizer)

    if verbose:
        print("Generation tokenizer configuration:")
        print(f"  Structured special tokens: {STRUCTURED_SPECIAL_TOKENS}")
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
