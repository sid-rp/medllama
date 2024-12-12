# training_utils.py
from datetime import datetime
from transformers import TrainingArguments
import wandb
from config import TrainingConfig

class TrainingUtils:
    @staticmethod
    def setup_wandb(config: TrainingConfig):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{config.finetuning_method}_{timestamp}_sid"
        logging_dir = f"log-{output_dir}"
        wandb.init(project="llama", name=output_dir, config=vars(config))
        return output_dir, logging_dir

    @staticmethod
    def create_training_args(config: TrainingConfig, output_dir: str, logging_dir: str) -> TrainingArguments:
        base_args = {
            "output_dir": output_dir,
            "eval_strategy": "epoch",
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_epochs": config.epochs,
            "weight_decay": config.weight_decay,
            "bf16": True,
            "logging_dir": logging_dir,
            "save_total_limit": config.save_total_limit,
            "learning_rate": config.learning_rate,
            "report_to": "wandb",
        }

        if config.finetuning_method == "qlora":
            base_args["optim"] = "paged_adamw_8bit"
        elif config.finetuning_method == "galore":
            base_args["optim"] = "galore_adamw"
            base_args["optim_target_modules"] = [
                "k_proj", "v_proj", "q_proj", "o_proj", 
                "gate_proj", "down_proj", "up_proj"
            ]

        return TrainingArguments(**base_args)

    @staticmethod
    def print_trainable_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")
