"""Language model for object extraction, correctness, and semantic clustering."""


def obtain_llm(name):
    if name not in {
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
    }:
        raise ValueError(f"Unsupported LLM: {name}")
    from llm.Qwen import Qwen

    return Qwen(name)
