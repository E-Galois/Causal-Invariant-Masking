"""Evaluate pure and SE-blended CIM scores from prepared shelves (CPU only)."""

import csv
import json
import shelve
from pathlib import Path

import numpy as np
from metric.AUROC import compute_auroc
from metric.AURAC import compute_aurac
from util.eed import NORMAL_SAMPLES, eed_mode, eed_score
from util.pipeline import (
    argument_parser,
    finalize_args,
    shelf_path,
    check_protocol,
)


def entropy(prob):
    return float(-sum(p * np.log2(p) for p in prob if not np.isclose(p, 0)))


def sd_score(record):
    original = np.array(record["orig_prob_CROP_se16vp16"])
    crop = np.array(record["vp_prob_CROP_se16vp16"])
    if original.shape != crop.shape or original.ndim != 1 or not len(original):
        raise ValueError("Invalid semantic distributions")
    uniform = np.ones(len(original)) / len(original)
    p = 0.99 * original + 0.01 * uniform
    q = 0.99 * crop + 0.01 * uniform
    return float(np.sum(q * np.log2(q / p)))


def evaluate(args, method):
    path = shelf_path(args, eed=method == "EED")
    rows = []
    with shelve.open(path, flag="r") as db:
        check_protocol(db, args)
        for idx in range(*args.subset):
            record = db[f"idx_{idx}"]
            label = 1.0 - float(record["flag_ans_correct"])
            se = entropy(record["orig_prob_CROP_se16vp16"])
            score = (
                sd_score(record)
                if method == "SD"
                else eed_score(record, eed_mode(args))
            )
            if label not in (0, 1) or not np.isfinite([se, score]).all():
                raise ValueError(f"Invalid label or score at idx_{idx}")
            rows.append([idx, label, se, score])
    values = np.asarray(rows)
    labels, se, pure = values[:, 1], values[:, 2], values[:, 3]
    mixed = (1 - args.lambda_value) * se + args.lambda_value * pure

    def summarize(scores):
        return {
            "AUROC": (
                compute_auroc(labels, scores) if len(np.unique(labels)) == 2 else None
            ),
            "AURAC": compute_aurac(labels, scores),
        }

    result = {
        "method": method,
        "shelf": path,
        "n": len(rows),
        "lambda": args.lambda_value,
        "SE16": summarize(se),
        "pure": summarize(pure),
        "blended": summarize(mixed),
    }
    if method == "EED":
        result["eed_mode"] = eed_mode(args)
        result["eed_answers_per_example"] = (
            1 if eed_mode(args) == "yes_no" else NORMAL_SAMPLES
        )
        result["eed_original_mode"] = 1 if eed_mode(args) == "yes_no" else 2
    if args.sweep_lambda:
        sweep = [
            {"lambda": float(weight), **summarize((1 - weight) * se + weight * pure)}
            for weight in np.linspace(0, 1, 40)
        ]
        result["exploratory_lambda_sweep"] = sweep
        if len(np.unique(labels)) == 2:
            result["exploratory_best"] = max(sweep, key=lambda row: row["AUROC"])
    output = Path(args.exp_dir) / "results"
    output.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mllm}_{args.benchmark}_{args.prefix}_{method}_{args.subset[0]}_{args.subset[1]}"
    if method == "EED":
        tag += f"_{eed_mode(args)}"
        if eed_mode(args) == "normal":
            tag += "_mode2"
    with (output / f"{tag}_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "error_label", "SE16", "pure", "blended"])
        writer.writerows([*row, float(blend)] for row, blend in zip(rows, mixed))
    with (output / f"{tag}.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def check_vp_prob_auroc_with_shelve(args):
    return evaluate(args, "SD")


def check_vp_emb_auroc_with_shelve(args):
    return evaluate(args, "EED")


def main():
    parser = argument_parser(__doc__)
    parser.add_argument("--method", choices=["SD", "EED", "both"], default="both")
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=0.05)
    parser.add_argument(
        "--sweep-lambda",
        action="store_true",
        help="Also report the legacy 40-weight sweep; best-on-test is exploratory",
    )
    args = finalize_args(parser.parse_args())
    if not 0 <= args.lambda_value <= 1:
        parser.error("--lambda must be between 0 and 1")
    if args.method in {"SD", "both"}:
        check_vp_prob_auroc_with_shelve(args)
    if args.method in {"EED", "both"}:
        check_vp_emb_auroc_with_shelve(args)


if __name__ == "__main__":
    main()
