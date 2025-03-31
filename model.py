import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LLMmodel:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None
        self.system_message = None
        self.user_message = None
        self.generation_params = kwargs

    def load_model(self, modelPath):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelPath,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config = quantization_config
        )
    
    def convert_chat_template(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        return input_ids, terminators

    def generate_answer(self):

        input_ids, terminators = self.convert_chat_template(self.system_message, self.user_message)
        outputs = self.model.generate(
            input_ids,
            eos_token_id=terminators,
            **self.generation_params
        )

        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        return response

    