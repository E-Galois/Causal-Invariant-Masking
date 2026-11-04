"""EED mode selection and the original sequence-level cosine-drift formulas."""

import numpy as np

NORMAL_SAMPLES = 8


def eed_mode(args):
    mode = getattr(args, "eed_mode", "auto")
    if mode == "auto":
        return "yes_no" if args.benchmark.startswith("POPE") else "normal"
    if mode not in {"yes_no", "normal"}:
        raise ValueError(f"Unknown EED mode: {mode}")
    return mode


def sequence_drift(label, expected):
    label, expected = np.asarray(label), np.asarray(expected)
    if (
        label.ndim != 2
        or label.shape != expected.shape
        or not label.size
        or not np.isfinite(label).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Invalid EED embeddings")
    dot = np.sum(label * expected, axis=1)
    denominator = np.linalg.norm(label, axis=1) * np.linalg.norm(expected, axis=1)
    value = np.sum(1 - dot / (denominator + 1e-8))
    if not np.isfinite(value):
        raise ValueError("Nonfinite EED drift")
    return value


def eed_score(record, mode):
    if mode == "yes_no":
        return float(
            sequence_drift(record["label_emb_vp"], record["prob_emb_orig_in_vp_out"])
        )
    if mode != "normal":
        raise ValueError(f"Unknown EED mode: {mode}")
    labels = record.get("label_emb_vp_samplings", [])
    expected = record.get("prob_emb_orig_in_vp_out_samplings", [])
    if len(labels) < NORMAL_SAMPLES or len(expected) < NORMAL_SAMPLES:
        raise ValueError(
            "EED normal mode needs the first eight cropped-answer embeddings; rerun prepare_eed.py to complete preparation"
        )
    # Mode 2: average eight sampled-answer sequence sums; exclude the single answer.
    values = [sequence_drift(labels[i], expected[i]) for i in range(NORMAL_SAMPLES)]
    return float(np.mean(values))


def eed_record_complete(record, mode):
    try:
        if mode == "yes_no" and "a_vp" not in record:
            return False
        eed_score(record, mode)
        return True
    except (KeyError, ValueError, TypeError):
        return False
