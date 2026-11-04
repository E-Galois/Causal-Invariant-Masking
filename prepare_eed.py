import copy
import shelve
from pathlib import Path
from util.pipeline import inference, apply_operation
from util.eed import NORMAL_SAMPLES, eed_mode, eed_record_complete, sequence_drift
from util.pipeline import (
    argument_parser,
    finalize_args,
    shelf_path,
    check_protocol,
    require_samples,
)


def get_prob_emb(args, data, mllm):
    hs = mllm.basic_forward(data=data)
    return (hs["prob_emb"], hs["label_emb"])


def pipeline_get_prob_emb_label_emb(
    args, idx, det, mllm, benchmark, llm, answer_checking_call, log_dict_cur
):
    print(f"======================== Index: {idx} ========================")
    data_orig = benchmark[idx]["data"]
    union_box = log_dict_cur["union_box"]
    operation = args.vp_op
    modified_image = apply_operation(data_orig["image"], operation, union_box)
    data_vp = copy.deepcopy(data_orig)
    data_vp["image"] = modified_image
    mode = eed_mode(args)
    if mode == "yes_no" and "a_vp" not in log_dict_cur:
        a_vp = inference(args, data_vp, mllm)["text"]
        log_dict_cur["a_vp"] = a_vp

    def embed_answer(answer):
        cropped = copy.deepcopy(data_vp)
        cropped["text"]["answer"] = answer
        original = copy.deepcopy(data_orig)
        original["text"]["answer"] = answer
        crop_prob, crop_label = get_prob_emb(args, cropped, mllm)
        orig_prob, orig_label = get_prob_emb(args, original, mllm)
        values = tuple(
            t.float().cpu().numpy()
            for t in (crop_prob, crop_label, orig_prob, orig_label)
        )
        sequence_drift(values[1], values[2])
        return values

    # Only mode 1 needs a separately generated single-answer pair.
    if mode == "yes_no" and not eed_record_complete(log_dict_cur, "yes_no"):
        values = embed_answer(log_dict_cur["a_vp"])
        for key, value in zip(
            (
                "prob_emb_vp",
                "label_emb_vp",
                "prob_emb_orig_in_vp_out",
                "label_emb_orig_in_vp_out",
            ),
            values,
        ):
            log_dict_cur[key] = value

    if mode == "normal":
        n_crop = args.sampling_time * args.sampling_time_outer_vp
        samples = log_dict_cur.get(f"a_vp_samplings_{args.vp_op}_{n_crop}", [])
        if len(samples) < NORMAL_SAMPLES:
            raise ValueError(
                "Normal EED needs eight saved cropped-view answers; complete SD sampling first"
            )
        # Lists use the same sample order as the original mode-2 implementation.
        labels = list(log_dict_cur.get("label_emb_vp_samplings", []))
        expected = list(log_dict_cur.get("prob_emb_orig_in_vp_out_samplings", []))
        for i in range(NORMAL_SAMPLES):
            try:
                sequence_drift(labels[i], expected[i])
                continue
            except (IndexError, ValueError, TypeError):
                pass
            _, crop_label, orig_prob, _ = embed_answer(samples[i])
            while len(labels) <= i:
                labels.append(None)
            while len(expected) <= i:
                expected.append(None)
            labels[i], expected[i] = crop_label, orig_prob
        log_dict_cur["label_emb_vp_samplings"] = labels
        log_dict_cur["prob_emb_orig_in_vp_out_samplings"] = expected


def main():
    args = finalize_args(
        argument_parser("Prepare CIM EED embeddings from an SD shelf").parse_args()
    )
    from factory.benchmark_factory import obtain_benchmark
    from factory.mllm_factory import obtain_mllm

    benchmark = obtain_benchmark(args)
    require_samples(args, benchmark)
    source, output = shelf_path(args), shelf_path(args, eed=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    mode = eed_mode(args)
    with shelve.open(source, flag="r") as src, shelve.open(output, flag="c") as db:
        check_protocol(src, args)
        check_protocol(db, args, initialize=True)
        # Keep SD labels/distributions alongside embeddings for identical evaluation.
        for idx in range(*args.subset):
            key = f"idx_{idx}"
            if key not in db:
                record = src[key]
                if not {
                    "flag_ans_correct",
                    "orig_prob_CROP_se16vp16",
                    "vp_prob_CROP_se16vp16",
                }.issubset(record):
                    raise ValueError(f"Incomplete SD preparation at {key}")
                db[key] = record
        pending = [
            i
            for i in range(*args.subset)
            if not eed_record_complete(db[f"idx_{i}"], mode)
        ]
        if pending:
            mllm = obtain_mllm(args.mllm)
            for idx in pending:
                record = db[f"idx_{idx}"]
                sample_key = f"a_vp_samplings_{args.vp_op}_{args.sampling_time * args.sampling_time_outer_vp}"
                if mode == "normal" and sample_key not in record:
                    record[sample_key] = src[f"idx_{idx}"][sample_key]
                pipeline_get_prob_emb_label_emb(
                    args, idx, None, mllm, benchmark, None, None, record
                )
                db[f"idx_{idx}"] = record
                db.sync()
    print(f"Saved CIM EED preparation ({mode}): {output}")


if __name__ == "__main__":
    main()
