import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import math
import warnings
from mllm.utils.internvl_conversation import get_conv_template

warnings.filterwarnings("ignore")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def build_transform(input_size):
    MEAN, STD = (IMAGENET_MEAN, IMAGENET_STD)
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            i % (target_width // image_size) * image_size,
            i // (target_width // image_size) * image_size,
            (i % (target_width // image_size) + 1) * image_size,
            (i // (target_width // image_size) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert("RGB")
    else:
        image = image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL:

    def __init__(self, version=None):
        if not version:
            self.version = "InternVL2-8B"
        else:
            self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"OpenGVLab/{self.version}"
        device_map = split_model(self.version)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

    def generate(self, data, temp, max_new_tokens=64):
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_k=None,
            top_p=None,
            output_scores=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict_in_generate=True,
        )
        pixel_values = load_image(data["image"], max_num=4).to(torch.bfloat16).to(0)
        question = data["text"]["question"]
        partial_answer = (
            data["text"]["partial_answer"] if "partial_answer" in data["text"] else None
        )
        outputs = self.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            answer_to_be_completed=partial_answer,
            raw=False,
        )
        res = {"text": outputs["response_text"]}
        return res

    def generate_multiple_answers(self, data, temp, num_samples=5, raw=False):
        """Sample answers from one image/question encoding; retain the original decoding convention."""
        if not temp > 0:
            print(
                "Warning: Temperature is not positive, so the generated answers may not be diverse."
            )
        if num_samples <= 0:
            return []
        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id
        device = self.model.device
        pixel_values = (
            load_image(data["image"], max_num=4).to(torch.bfloat16).to(device)
        )
        question = data["text"]["question"]
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        if not raw:
            if "<image>" not in question:
                question = "<image>\n" + question
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
        else:
            query = question
        num_patches = pixel_values.shape[0] if pixel_values is not None else 0
        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches
            + "</img>"
        )
        final_query = query.replace("<image>", image_tokens, 1)
        model_inputs = self.tokenizer(final_query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        generation_config = dict(
            max_new_tokens=64,
            do_sample=True,
            temperature=temp,
            top_k=None,
            top_p=None,
            num_return_sequences=num_samples,
        )
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config["eos_token_id"] = eos_token_id
        with torch.no_grad():
            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                begin_suppress_tokens=[eos_token_id],
                **generation_config,
            )
        output_ids = generation_output
        answers = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return answers

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        answer_to_be_completed=None,
        raw=False,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):
        if history is None and pixel_values is not None and ("<image>" not in question):
            question = "<image>\n" + question
        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        if raw:
            query = question
        else:
            template.append_message(template.roles[0], question)
            if answer_to_be_completed is not None:
                template.append_message(
                    template.roles[1], (answer_to_be_completed, False)
                )
            else:
                template.append_message(template.roles[1], None)
            query = template.get_prompt()
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")
        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        model_inputs = tokenizer(query, return_tensors="pt")
        device = torch.device(
            self.model.language_model.device if torch.cuda.is_available() else "cpu"
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response_ids = generation_output.sequences
        prompt_ids_length = input_ids.shape[1]
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        results = {
            "input_ids": input_ids,
            "response_text": response,
            "prompt_text": query,
            "prompt_ids_length": prompt_ids_length,
            "response_ids": response_ids,
            "hidden_states": generation_output.get("hidden_states", None),
            "scores": generation_output.get("scores", None),
            "attentions": generation_output.get("attentions", None),
            "raw_output": generation_output,
        }
        if return_history:
            return (results, history)
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            if verbose:
                print(query_to_print, results)
            return results

    def basic_forward(self, data):
        """Teacher-force the answer; return token expected embeddings and label embeddings."""
        device = torch.device(
            self.model.language_model.device if torch.cuda.is_available() else "cpu"
        )
        pixel_values = (
            load_image(data["image"], max_num=4).to(torch.bfloat16).to(device)
        )
        question = data["text"]["question"]
        answer = data["text"]["answer"]
        if "<image>" not in question:
            question = "<image>\n" + question
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        prompt_question_only = template.get_prompt()
        template.messages.pop()
        template.append_message(template.roles[1], answer)
        prompt_full = template.get_prompt()
        num_patches = pixel_values.shape[0] if pixel_values is not None else 0
        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches
            + "</img>"
        )
        final_prompt_question_only = prompt_question_only.replace(
            "<image>", image_tokens, 1
        )
        final_prompt_full = prompt_full.replace("<image>", image_tokens, 1)
        question_input_ids = self.tokenizer(
            final_prompt_question_only, return_tensors="pt"
        ).input_ids
        last_before_gen_pos = question_input_ids.shape[1] - 1
        full_inputs = self.tokenizer(final_prompt_full, return_tensors="pt")
        full_input_ids = full_inputs["input_ids"].to(device)
        attention_mask = full_inputs["attention_mask"].to(device)
        before_eos_pos = full_input_ids.shape[1] - 2
        image_flags = torch.ones(num_patches, 1, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = self.model.forward(
                pixel_values=pixel_values,
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                image_flags=image_flags,
            )
        start_logit_idx = last_before_gen_pos
        end_logit_idx = before_eos_pos
        answer_logits = outputs.logits[:, start_logit_idx:end_logit_idx, :]
        answer_labels = full_input_ids[:, start_logit_idx + 1 : end_logit_idx + 1]
        answer_probs = torch.nn.functional.softmax(answer_logits, dim=-1)
        W_emb = self.model.get_input_embeddings().weight.detach()
        prob_emb = (
            torch.matmul(answer_probs.to(W_emb.dtype), W_emb).squeeze(dim=0).float()
        )
        label_emb = W_emb[answer_labels].squeeze(dim=0).float()
        res = {"prob_emb": prob_emb, "label_emb": label_emb}
        return res
