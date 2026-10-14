from transformers import AutoTokenizer


def validate_speculative_vocab(target_tokenizer, draft_tokenizer,
                               target_vocab_size, draft_vocab_size):
    """Token-level verification needs identical IDs, not only equal sizes."""
    if target_vocab_size != draft_vocab_size:
        raise ValueError(
            "Speculative decoding requires equal target/draft vocabulary "
            f"dimensions; got {target_vocab_size} and {draft_vocab_size}")
    target_vocab = target_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    if target_vocab != draft_vocab:
        raise ValueError(
            "Speculative decoding requires identical token-to-ID mappings; "
            "equal vocab_size or model family alone is insufficient")
    if any(i < 0 or i >= target_vocab_size for i in target_vocab.values()):
        raise ValueError("Tokenizer contains IDs outside the model vocabulary")


# Infer model family based on model path name
def infer_model_family(model_path: str) -> str:
        """Infer if model is Llama or Qwen based on path name.

        ``qwama`` is mapped to ``qwen`` because turboderp/Qwama-* is a Qwen2
        architecture (Llama-3 vocab transplanted). The architecture-aware
        downstream code (model_runner dispatch, llm_engine cross-family
        vocab-match override) handles the actual cross-family pairing.
        """
        model_path_lower = model_path.lower()
        if "qwama" in model_path_lower:
            return "qwen"
        if "llama" in model_path_lower:
            return "llama"
        elif "qwen" in model_path_lower:
            return "qwen"
        else:
            return "unknown"


def decode_tokens(token_ids: list[int], tokenizer: AutoTokenizer) -> list[str]:
    decoded = []
    for token in token_ids:
        try:
            text = tokenizer.decode([token], skip_special_tokens=False)
            decoded.append(text)
        except Exception:
            decoded.append(f"<token_id:{token}>")
    return decoded
