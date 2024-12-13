import sys
import json
import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    BitsAndBytesConfig
)
from .handler import DataHandler

assert torch.cuda.is_available(), "No cuda device detected"

class Inferer:
    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        base_model: str = "meta-llama/Llama-3.2-3B",
        model_max_length: int = 256,
        finetuning_method: str = "sft"
    ) -> None:
        self.model = self._load_model(
            model_name=model_name,
            base_model=base_model,
            finetuning_method=finetuning_method
        )

        tokenizer = self._load_tokenizer(base_model)
        
        self.data_handler = DataHandler(
            tokenizer,
            prompt_template=prompt_template,
            model_max_length=model_max_length,
            train_on_inputs=False,
        )

    def _load_model(
        self,
        model_name: str,
        base_model: str,
        finetuning_method: str
    ) -> torch.nn.Module:
        if finetuning_method == "qlora":
            # QLoRA specific configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map={"": 0}
            )
            model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(
                model,
                model_name,    # <- Changed model_id to just model_name
                is_trainable=False,
                device_map={"": 0},
            )
        
        elif finetuning_method == "lora":
            # LoRA configuration
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.bfloat16,
                device_map={"": 0}
            )
            model = PeftModel.from_pretrained(
                model,
                model_id=model_name,
                device_map={"": 0},
            )
        
        elif finetuning_method == "galore":
            # Galore configuration
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": 0}
            )
            # model = PeftModel.from_pretrained(
            #     model,
            #     model_id=model_name,
            #     device_map={"": 0},
            # )
        
        else:  # sft
            # Standard fine-tuning configuration
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": 0}
            )

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        return model

    def _load_tokenizer(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        return tokenizer

    def __call__(
        self,
        input: str,
        instruction: str = None,
        output: str = None,
        max_new_tokens: int = 128,
        verbose: bool = False,
        **generation_kwargs,
    ) -> str:
        prompt = self.data_handler.generate_prompt(
            instruction=instruction,
            input=input,
            output=output
        )
        
        if verbose:
            print(prompt)

        input_tokens = self.data_handler.tokenizer(prompt, return_tensors="pt", padding=True,
        return_attention_mask=True)
        input_token_ids = input_tokens["input_ids"].to("cuda")

        generation_config = GenerationConfig(**generation_kwargs)

        with torch.no_grad():
            generation_output = self.model.generate(
                pad_token_id=self.data_handler.tokenizer.eos_token_id,
                input_ids=input_token_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        generation_output_decoded = self.data_handler.tokenizer.decode(generation_output.sequences[0])
        split = f'{self.data_handler.prompt_template["output"]}{output or ""}'
        response = generation_output_decoded.split(split)[-1].strip()
        return response