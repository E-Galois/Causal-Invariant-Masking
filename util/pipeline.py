"""Shared CLI, crop geometry, and resumable storage for the main pipeline."""

import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SIZES = {
    "VQAv2": 2000,
    "OKVQA": 2000,
    "AdVQA": 2000,
    "POPE_rand": 3000,
    "POPE_pop": 3000,
    "POPE_adv": 3000,
    "MME": 2374,
}


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def apply_operation(image, operation, box):
    if operation != "CROP":
        raise ValueError("The main CIM pipeline supports CROP only")
    return image.crop((box.xmin, box.ymin, box.xmax, box.ymax))


def inference(args, data, mllm):
    return mllm.generate(data=data, temp=args.inference_temp)


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mllm", choices=["InternVL", "QwenVL"], required=True)
    parser.add_argument("--benchmark", choices=list(BENCHMARK_SIZES), required=True)
    parser.add_argument("--exp_dir", default=str(ROOT / "exp_cim"))
    parser.add_argument("--prefix", default="CIM")
    parser.add_argument("--subset", type=int, nargs=2)
    parser.add_argument("--vp_op", choices=["CROP"], default="CROP")
    parser.add_argument("--inference_temp", type=float, default=0.1)
    parser.add_argument("--sampling_temp", type=float, default=0.5)
    parser.add_argument("--sampling_time", type=int, default=16)
    parser.add_argument("--sampling_time_outer_orig", type=int, default=2)
    parser.add_argument("--sampling_time_outer_vp", type=int, default=1)
    parser.add_argument("--llm", default="Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--eed-mode",
        choices=["auto", "yes_no", "normal"],
        default="auto",
        help="EED: auto selects yes_no for POPE, normal elsewhere",
    )
    return parser


def finalize_args(args):
    if args.subset is None:
        args.subset = [0, BENCHMARK_SIZES[args.benchmark]]
    if not 0 <= args.subset[0] < args.subset[1]:
        raise ValueError("--subset must specify a nonempty interval [start, end)")
    if (
        args.sampling_time <= 0
        or args.sampling_time_outer_orig <= 0
        or args.sampling_time_outer_vp <= 0
    ):
        raise ValueError("Sampling counts must be positive")
    if (
        min(
            args.sampling_time * args.sampling_time_outer_orig,
            args.sampling_time * args.sampling_time_outer_vp,
        )
        < 16
    ):
        raise ValueError("The main SD calculation needs at least 16 samples per view")
    if min(args.inference_temp, args.sampling_temp) <= 0:
        raise ValueError("Generation temperatures must be positive")
    if not args.prefix or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.prefix
    ):
        raise ValueError(
            "prefix must contain only letters, digits, underscores, or hyphens"
        )
    return args


def shelf_path(args, eed=False):
    suffix = "__eed" if eed else ""
    return str(
        Path(args.exp_dir)
        / "arch"
        / f"log_{args.mllm}_{args.benchmark}_{args.prefix}{suffix}.db"
    )


def protocol(args):
    return {
        "mllm": args.mllm,
        "benchmark": args.benchmark,
        "prompt": (
            "vs"
            if "QwenVL" in args.mllm and args.benchmark in ["VQAv2", "OKVQA"]
            else "default"
        ),
        "operation": args.vp_op,
        "inference_temp": args.inference_temp,
        "sampling_temp": args.sampling_temp,
        "llm": args.llm,
        "n_original": args.sampling_time * args.sampling_time_outer_orig,
        "n_crop": args.sampling_time * args.sampling_time_outer_vp,
    }


def check_protocol(db, args, initialize=False):
    expected = protocol(args)
    if "protocol" not in db:
        if not initialize or len(db):
            raise ValueError(
                "Shelf has no release protocol metadata; use a fresh CIM shelf"
            )
        db["protocol"] = expected
    elif db["protocol"] != expected:
        raise ValueError(
            f"Shelf protocol mismatch: saved={db['protocol']}, requested={expected}"
        )


def require_samples(args, benchmark):
    if args.subset[1] > len(benchmark):
        raise ValueError(
            f"Requested end {args.subset[1]} exceeds dataset size {len(benchmark)}"
        )
