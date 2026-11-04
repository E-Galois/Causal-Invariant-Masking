"""Datasets and model-dependent question prompts for CIM."""


def obtain_benchmark(args):
    b_kwargs = (
        {"prompt": "vs"}
        if "QwenVL" in args.mllm and args.benchmark in ["VQAv2", "OKVQA"]
        else {}
    )
    if args.benchmark == "VQAv2":
        from benchmark.comprehension.VQAv2 import VQAv2

        return VQAv2(**b_kwargs)
    if args.benchmark == "OKVQA":
        from benchmark.comprehension.OKVQA import OKVQA

        return OKVQA(**b_kwargs)
    if args.benchmark == "AdVQA":
        from benchmark.comprehension.AdVQA import AdVQA

        return AdVQA(**b_kwargs)
    if args.benchmark.startswith("POPE_"):
        from benchmark.comprehension.POPE import POPE

        split = {
            "POPE_rand": "random",
            "POPE_pop": "popular",
            "POPE_adv": "adversarial",
        }
        if args.benchmark in split:
            return POPE(key=split[args.benchmark], **b_kwargs)
    if args.benchmark == "MME":
        from benchmark.comprehension.MME import MME

        return MME(**b_kwargs)
    raise ValueError(f"Unsupported benchmark: {args.benchmark}")
