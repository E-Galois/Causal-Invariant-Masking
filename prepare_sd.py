import copy
import gc
import shelve
from pathlib import Path
from json_repair import repair_json
from util.pipeline import inference, apply_operation, BoundingBox
from util.pipeline import (
    argument_parser,
    finalize_args,
    shelf_path,
    check_protocol,
    require_samples,
)


def sampling_parallel(args, data, mllm, num_samples):
    ans_list = mllm.generate_multiple_answers(
        data=data, temp=args.sampling_temp, num_samples=num_samples
    )
    return ans_list


def get_union_box(detections):
    """Calculates the union bounding box of all detections."""
    if not detections:
        return None
    boxes = [d.box for d in detections]
    xmin = min((box.xmin for box in boxes))
    ymin = min((box.ymin for box in boxes))
    xmax = max((box.xmax for box in boxes))
    ymax = max((box.ymax for box in boxes))
    return BoundingBox(xmin, ymin, xmax, ymax)


def pipeline_get_initials_det_mllm_llm(
    args, idx, det, mllm, benchmark, llm, answer_checking_call, log_dict_cur
):
    print(f"======================== Index: {idx} ========================")
    data_orig = benchmark[idx]["data"]
    q_orig = data_orig["text"]["question"]
    a_orig = inference(args, data_orig, mllm)["text"]
    print(f"[orig] {q_orig} -> {a_orig}")
    log_dict_cur["q_orig"] = q_orig
    log_dict_cur["a_orig"] = a_orig
    object_prompt = 'What are the key objects in this question-answer pair?\nQ: %s\nA: %s\nOutput in json format: {"key_objects": ["key object 1", "key object 2", ...]}'
    object_prompt = object_prompt % (
        data_orig["text"]["question"],
        log_dict_cur["a_orig"],
    )
    object_name_raw = llm.generate(object_prompt, temp=args.inference_temp)
    decoded = repair_json(object_name_raw, return_objects=True)
    object_name_decoded = None
    if isinstance(decoded, list):
        for obj in decoded:
            if isinstance(obj, dict) and "key_objects" in obj:
                object_name_decoded = obj
    elif isinstance(decoded, dict) and "key_objects" in decoded:
        object_name_decoded = decoded
    else:
        raise ValueError("key objects cannot be decoded")
    if object_name_decoded is None:
        raise ValueError("key objects cannot be decoded")
    object_name = object_name_decoded["key_objects"]
    if isinstance(object_name, str):
        object_name = [object_name]
    log_dict_cur["object_name"] = object_name
    if len(object_name) == 0:
        detections = None
    else:
        detections = det.detect(data_orig["image"], object_name, threshold=0.3)
    log_dict_cur["detections"] = detections
    width, height = data_orig["image"].size
    if not detections:
        print(
            f"[WARNING]: GroundingDINO could not find '{object_name}'; using the center box."
        )
        union_box = BoundingBox(
            width // 4, height // 4, 3 * width // 4, 3 * height // 4
        )
        log_dict_cur["no_valid_detections"] = True
    else:
        filtered_detections = []
        for detection in detections:
            if "image" in detection.label.lower():
                box_area = (detection.box.xmax - detection.box.xmin) * (
                    detection.box.ymax - detection.box.ymin
                )
                image_area = width * height
                if box_area > 0.95 * image_area:
                    print(f"[WARNING]: Filtered class <{detection.label}>.")
                    continue
            filtered_detections.append(detection)
        if not filtered_detections:
            print(
                "[WARNING]: After filtering, no valid detections left. Using center bounding box."
            )
            union_box = BoundingBox(
                width // 4, height // 4, 3 * width // 4, 3 * height // 4
            )
            log_dict_cur["no_valid_detections"] = True
        else:
            union_box = get_union_box(filtered_detections)
            log_dict_cur["no_valid_detections"] = False
    log_dict_cur["union_box"] = union_box


def pipeline_mllm_sample_orig_and_vp(
    args, idx, det, mllm, benchmark, llm, answer_checking_call, log_dict_cur
):
    print(f"======================== Index: {idx} ========================")
    data_orig = benchmark[idx]["data"]
    a_orig_samplings = []
    for i_sampling_outer in range(args.sampling_time_outer_orig):
        a_orig_samplings.extend(
            sampling_parallel(args, data_orig, mllm, args.sampling_time)
        )
    for i, a_orig_sampling in enumerate(a_orig_samplings):
        print(f"[{i}] {a_orig_sampling}")
    n_sample_orig = args.sampling_time * args.sampling_time_outer_orig
    log_dict_cur[f"a_orig_samplings_{n_sample_orig}"] = a_orig_samplings
    union_box = log_dict_cur["union_box"]
    operation = args.vp_op
    modified_image = apply_operation(data_orig["image"], operation, union_box)
    data_vp = {"image": modified_image}
    data_vp["text"] = data_orig["text"]
    a_vp_samplings = []
    for i_sampling_outer in range(args.sampling_time_outer_vp):
        a_vp_samplings.extend(
            sampling_parallel(args, data_vp, mllm, args.sampling_time)
        )
    for i, a_vp_sampling in enumerate(a_vp_samplings):
        print(f"[{i}] {a_vp_sampling}")
    n_sample_vp = args.sampling_time * args.sampling_time_outer_vp
    log_dict_cur[f"a_vp_samplings_{args.vp_op}_{n_sample_vp}"] = a_vp_samplings


def pipeline_llm_acc_and_semantic_clusters(
    args, idx, det, mllm, benchmark, llm, answer_checking_call, log_dict_cur
):
    print(f"======================== Index: {idx} ========================")
    data_orig = benchmark[idx]["data"]
    a_orig = log_dict_cur["a_orig"]
    ans = {"text": a_orig}
    flag_ans_correct = answer_checking_call(args, ans, data_orig, llm)
    log_dict_cur["flag_ans_correct"] = flag_ans_correct
    n_sample_orig = args.sampling_time * args.sampling_time_outer_orig
    a_orig_samplings = log_dict_cur[f"a_orig_samplings_{n_sample_orig}"]
    n_sample_vp = args.sampling_time * args.sampling_time_outer_vp
    a_vp_samplings = log_dict_cur[f"a_vp_samplings_{args.vp_op}_{n_sample_vp}"]
    n_use_orig = 16
    n_use_vp = 16
    ci_orig, ci_vp, orig_cnt, vp_cnt, orig_prob, vp_prob = (
        get_orig_vp_dual_prob_distribution_parallel(
            args,
            data_orig["text"]["question"],
            a_orig_samplings[:n_use_orig],
            a_vp_samplings[:n_use_vp],
            llm,
        )
    )
    log_dict_cur[f"ci_orig_{args.vp_op}_se{n_use_orig}vp{n_use_vp}"] = ci_orig
    log_dict_cur[f"ci_vp_{args.vp_op}_se{n_use_orig}vp{n_use_vp}"] = ci_vp
    log_dict_cur[f"orig_cnt_{args.vp_op}_se{n_use_orig}vp{n_use_vp}"] = orig_cnt
    log_dict_cur[f"vp_cnt_{args.vp_op}_se{n_use_orig}vp{n_use_vp}"] = vp_cnt
    log_dict_cur[f"orig_prob_{args.vp_op}_se{n_use_orig}vp{n_use_vp}"] = orig_prob
    log_dict_cur[f"vp_prob_{args.vp_op}_se{n_use_orig}vp{n_use_vp}"] = vp_prob
    print(f"  [ci_orig] : {ci_orig}")
    print(f"  [ci_vp]   : {ci_vp}")
    print(f"  [orig_cnt] : {orig_cnt}")
    print(f"  [vp_cnt]   : {vp_cnt}")
    print(f"  [orig_prob]: {orig_prob}")
    print(f"  [vp_prob]  : {vp_prob}")


def format_prompt(q_str, ans_str, gt_str):
    """Helper just to return string, not call LLM"""
    return f"Question: {{{q_str}}} True Answer: {{{gt_str}}} Is the following answer correct? Answer: {{{ans_str}}} Respond with 'no' if this is totally different from the True Answer. Respond with 'yes' when they are of the same meaning."


def get_orig_vp_dual_prob_distribution_parallel(
    args, question, t_list_orig, t_list_vp, llm
):
    t_list = copy.deepcopy(t_list_orig)
    t_list.extend(t_list_vp)
    ans_cluster_idx = [-1] * len(t_list)
    cur_cluster_idx = 0
    for i in range(len(t_list)):
        if ans_cluster_idx[i] == -1:
            ans_cluster_idx[i] = cur_cluster_idx
            candidates_indices = []
            prompts = []
            for j in range(i + 1, len(t_list)):
                if ans_cluster_idx[j] == -1:
                    candidates_indices.append(j)
                    p = format_prompt(question, t_list[i], t_list[j])
                    prompts.append(p)
            if prompts:
                responses = llm.generate(prompts, temp=0.1)
                for idx_in_batch, resp in enumerate(responses):
                    if "yes" in resp.lower():
                        original_idx = candidates_indices[idx_in_batch]
                        ans_cluster_idx[original_idx] = cur_cluster_idx
            cur_cluster_idx += 1
    ci_orig = ans_cluster_idx[: len(t_list_orig)]
    ci_vp = ans_cluster_idx[len(t_list_orig) :]
    num_clusters = cur_cluster_idx
    orig_count_dist = [0] * num_clusters
    vp_count_dist = [0] * num_clusters
    for ci in ci_orig:
        if ci == -1:
            raise ValueError("Cluster index cannot be -1")
        orig_count_dist[ci] += 1
    for ci in ci_vp:
        if ci == -1:
            raise ValueError("Cluster index cannot be -1")
        vp_count_dist[ci] += 1
    total_orig = len(ci_orig)
    total_vp = len(ci_vp)
    orig_prob_dist = [count / total_orig for count in orig_count_dist]
    vp_prob_dist = [count / total_vp for count in vp_count_dist]
    return (
        ci_orig,
        ci_vp,
        orig_count_dist,
        vp_count_dist,
        orig_prob_dist,
        vp_prob_dist,
    )


def main():
    args = finalize_args(
        argument_parser(
            "Prepare CIM answers, crops, samples, and semantic distributions"
        ).parse_args()
    )
    from factory.benchmark_factory import obtain_benchmark
    from factory.mllm_factory import obtain_mllm
    from factory.llm_factory import obtain_llm
    from gsa.gounded_sam import GroundingDINO
    from util.llm_checkings import answer_checking_common, answer_checking_okvqa
    import torch

    benchmark = obtain_benchmark(args)
    require_samples(args, benchmark)
    checking = (
        answer_checking_okvqa if args.benchmark == "OKVQA" else answer_checking_common
    )
    path = shelf_path(args)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    indices = range(*args.subset)
    with shelve.open(path, flag="c") as db:
        check_protocol(db, args, initialize=True)

        # Stage 1: original answer and answer-conditioned RoI.
        pending = [
            i
            for i in indices
            if not {"q_orig", "a_orig", "union_box"}.issubset(db.get(f"idx_{i}", {}))
        ]
        if pending:
            det, mllm, llm = (
                GroundingDINO(),
                obtain_mllm(args.mllm),
                obtain_llm(args.llm),
            )
            for idx in pending:
                record = db.get(f"idx_{idx}", {})
                pipeline_get_initials_det_mllm_llm(
                    args, idx, det, mllm, benchmark, llm, checking, record
                )
                db[f"idx_{idx}"] = record
                db.sync()
            del det, mllm, llm
            gc.collect()
            torch.cuda.empty_cache()

        # Stage 2: independent original-view and crop-view samples.
        sample_keys = {
            f"a_orig_samplings_{args.sampling_time * args.sampling_time_outer_orig}",
            f"a_vp_samplings_CROP_{args.sampling_time * args.sampling_time_outer_vp}",
        }
        pending = [i for i in indices if not sample_keys.issubset(db[f"idx_{i}"])]
        if pending:
            mllm = obtain_mllm(args.mllm)
            for idx in pending:
                record = db[f"idx_{idx}"]
                pipeline_mllm_sample_orig_and_vp(
                    args, idx, None, mllm, benchmark, None, checking, record
                )
                db[f"idx_{idx}"] = record
                db.sync()
            del mllm
            gc.collect()
            torch.cuda.empty_cache()

        # Stage 3: correctness labels and joint semantic clustering (16+16).
        metric_keys = {
            "flag_ans_correct",
            "orig_prob_CROP_se16vp16",
            "vp_prob_CROP_se16vp16",
        }
        pending = [i for i in indices if not metric_keys.issubset(db[f"idx_{i}"])]
        if pending:
            llm = obtain_llm(args.llm)
            for idx in pending:
                record = db[f"idx_{idx}"]
                pipeline_llm_acc_and_semantic_clusters(
                    args, idx, None, None, benchmark, llm, checking, record
                )
                db[f"idx_{idx}"] = record
                db.sync()
    print(f"Saved CIM preparation: {path}")


if __name__ == "__main__":
    main()
