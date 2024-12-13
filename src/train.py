import torch
import wandb
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from config import TrainingConfig
from model_factory import ModelFactory
from training_utils import TrainingUtils
from handler import DataHandler

def main():
    # Initialize configuration
    config = TrainingConfig.from_args()
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Initialize data handler and load dataset
    data_handler = DataHandler(
        tokenizer=tokenizer,
        prompt_template=config.prompt_template,
        model_max_length=config.model_max_length,
        train_on_inputs=config.train_on_inputs,
    )

    # Load and process dataset
    data = load_dataset("json", data_files=config.data_path)
    if config.val_set_size > 0:
        data = (
            data["train"]
            .train_test_split(test_size=config.val_set_size, shuffle=True, seed=42)
            .map(data_handler.generate_and_tokenize_prompt)
        )
    else:
        data = data.shuffle(seed=42).map(data_handler.generate_and_tokenize_prompt)

    # Setup training environment
    output_dir, logging_dir = TrainingUtils.setup_wandb(config)
    
    # Initialize model
    model = ModelFactory.create_model(config)
    model.config.use_cache = False
    
    # Print model parameters
    TrainingUtils.print_trainable_parameters(model)
    
    # Setup training arguments and trainer
    training_args = TrainingUtils.create_training_args(config, output_dir, logging_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"] if config.val_set_size > 0 else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    # Setup optimizer and scheduler for SFT and LoRA
    if config.finetuning_method in ["sft", "lora"]:
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        num_training_steps = (
            config.epochs * len(data["train"]) 
            // (config.batch_size * training_args.gradient_accumulation_steps)
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )
        trainer.optimizers = (optimizer, scheduler)

    # Train and save model
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete. Model and tokenizer saved at:", output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()