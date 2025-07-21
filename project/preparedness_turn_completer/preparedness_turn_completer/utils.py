# Module-level constant for context window lengths
CONTEXT_WINDOW_LENGTHS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "o1-mini": 128_000,
    "o1-mini-2024-09-12": 128_000,
    "o1": 200_000,
    "o1-2024-12-17": 200_000,
    "o3": 200_000,
    "o3-mini-2025-01-31": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "o4-mini-deep-research-2025-06-26": 200_000,
    "o4-mini-deep-research": 200_000,
    "o3-deep-research-2025-06-26": 200_000,
    "o3-deep-research": 200_000,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1": 1_047_576,
    "o1-preview": 128_000,
    "gpt-4-turbo": 128_000,
}


def get_model_context_window_length(model: str) -> int:
    if model not in CONTEXT_WINDOW_LENGTHS:
        raise ValueError(f"Model {model} not found in context window lengths")
    return CONTEXT_WINDOW_LENGTHS[model]
