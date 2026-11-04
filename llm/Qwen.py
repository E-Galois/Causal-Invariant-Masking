import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class Qwen:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"Qwen/{self.version}"
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.use_flash_attn = True
        config.bf16 = True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, question, temp: float):
        if isinstance(question, str):
            questions = [question]
            is_single_question = True
        else:
            questions = question
            is_single_question = False
        all_messages = []
        for q in questions:
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": q},
            ]
            all_messages.append(messages)
        texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in all_messages
        ]
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=temp,
            top_p=0.8,
            repetition_penalty=1.05,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if is_single_question:
            return response[0]
        else:
            return response
