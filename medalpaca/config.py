# config.py
from dataclasses import dataclass
from typing import Optional
from argparse import ArgumentParser

@dataclass
class TrainingConfig:
    prompt_template: str
    model_max_length: int
    train_on_inputs: bool
    data_path: str
    val_set_size: float
    batch_size: int
    epochs: int
    learning_rate: float
    finetuning_method: str
    model_name: str = "meta-llama/Llama-3.2-3B"
    gradient_accumulation_steps: int = 6
    weight_decay: float = 0.01
    save_total_limit: int = 2

    @classmethod
    def from_args(cls):
        parser = ArgumentParser()
        parser.add_argument("--prompt_template", type=str, default="medalpaca/prompt_templates/medalpaca_new.json")
        parser.add_argument("--model_max_length", type=int, default=256)
        parser.add_argument("--train_on_inputs", type=bool, default=True)
        parser.add_argument("--data_path", type=str, default="/scratch/sp7835/medAlpaca/data/merged_medical_meadow.json")
        parser.add_argument("--val_set_size", type=float, default=0.1)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--finetuning_method", type=str, choices=["sft", "lora", "qlora", "galore"], required=True)
        
        args = parser.parse_args()
        return cls(**vars(args))
