# model_factory.py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import TrainingConfig

class ModelFactory:
    @staticmethod
    def create_model(config: TrainingConfig):
        if config.finetuning_method == "qlora":
            return ModelFactory._create_qlora_model(config)
        elif config.finetuning_method == "lora":
            return ModelFactory._create_lora_model(config)
        elif config.finetuning_method == "galore":
            return ModelFactory._create_galore_model(config)
        else:
            return ModelFactory._create_sft_model(config)

    @staticmethod
    def _create_qlora_model(config):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(config.model_name, quantization_config=bnb_config)
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        qlora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["k_proj", "v_proj", "q_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, qlora_config)

    @staticmethod
    def _create_lora_model(config):
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["k_proj", "v_proj", "q_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.gradient_checkpointing_enable()
        return get_peft_model(model, lora_config)

    @staticmethod
    def _create_galore_model(config):
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
        model.gradient_checkpointing_enable()
        return model

    @staticmethod
    def _create_sft_model(config):
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
        model.gradient_checkpointing_enable()
        return model
