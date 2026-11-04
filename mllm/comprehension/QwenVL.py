import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class QwenVL:

    def __init__(self, version=None):
        if not version:
            self.version = "Qwen2.5-VL-7B-Instruct"
        else:
            self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"Qwen/{self.version}"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(f"Qwen/{self.version}")

    def generate(self, data, temp, max_new_tokens=64):
        image_input = data["image"]
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(
                "image_input must be a file path (str) or a PIL Image object"
            )
        question = data["text"]["question"]
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}, {"type": "image"}],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(
            0, torch.float16
        )
        outputs = self.model.generate(
            **inputs,
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
        ans_text = self.processor.decode(
            outputs["sequences"][0, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        res = {"text": ans_text}
        return res

    def generate_multiple_answers(self, data, temp, num_samples=5):
        image_input = data["image"]
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(
                "image_input must be a file path (str) or a PIL Image object"
            )
        question = data["text"]["question"]
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}, {"type": "image"}],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(
            0, torch.float16
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=temp,
            top_k=None,
            top_p=None,
            num_return_sequences=num_samples,
        )
        answers = self.processor.batch_decode(
            outputs[:, len(inputs.input_ids[0]) :], skip_special_tokens=True
        )
        answers = [ans_text.split("ASSISTANT: ")[-1].strip() for ans_text in answers]
        return answers

    def basic_forward(self, data):
        """Teacher-force the answer; return token expected embeddings and label embeddings."""
        device = self.model.device
        image_input = data["image"]
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(
                "image_input must be a file path (str) or a PIL Image object"
            )
        question = data["text"]["question"]
        answer = data["text"]["answer"]
        conversation_context = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text_context = self.processor.apply_chat_template(
            conversation_context, add_generation_prompt=True
        )
        conversation_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        text_full = self.processor.apply_chat_template(
            conversation_full, add_generation_prompt=False
        )
        inputs_context = self.processor(
            images=image, text=text_context, return_tensors="pt"
        )
        last_before_gen_pos = inputs_context["input_ids"].shape[1] - 1
        full_inputs = self.processor(images=image, text=text_full, return_tensors="pt")
        full_inputs = full_inputs.to(device)
        full_input_ids = full_inputs["input_ids"]
        before_eos_pos = full_input_ids.shape[1] - 2
        with torch.no_grad():
            outputs = self.model(
                **full_inputs, output_hidden_states=False, return_dict=True
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
