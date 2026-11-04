"""The two MLLMs used for the main results."""


def obtain_mllm(name):
    if name == "InternVL":
        from mllm.comprehension.InternVL import InternVL

        return InternVL()
    if name == "QwenVL":
        from mllm.comprehension.QwenVL import QwenVL

        return QwenVL()
    raise ValueError(f"Unsupported MLLM: {name}; choose InternVL or QwenVL")
