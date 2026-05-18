from transformers import AutoTokenizer


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
