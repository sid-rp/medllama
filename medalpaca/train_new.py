import os
import sys
from datetime import datetime
from argparse import ArgumentParser
from typing import Tuple, Union
import wandb
import torch
from torch.optim import AdamW
from datasets import load_dataset
from handler import DataHandler
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--prompt_template", type=str, default="medalpaca/prompt_templates/medalpaca.json")
parser.add_argument("--model_max_length", type=int, default=256)
parser.add_argument("--train_on_inputs", type=bool, default=True)
parser.add_argument("--data_path", type=str, default="/scratch/sp7835/medAlpaca/data/merged_medical_meadow.json")
parser.add_argument("--val_set_size", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--finetuning_method", type=str, choices=["sft", "lora", "qlora", "galore"], required=True)

args = parser.parse_args()

# Define constants and configurations
model_name = "meta-llama/Llama-3.2-3B"
finetuning_method = args.finetuning_method
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"{finetuning_method}_{timestamp}_sid"
logging_dir = f"log-{output_dir}"
wandb.init(project="llama", name=output_dir, config=args)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

# Initialize data handler
data_handler = DataHandler(
    tokenizer=tokenizer,
    prompt_template=args.prompt_template,
    model_max_length=args.model_max_length,
    train_on_inputs=args.train_on_inputs,
)

# Load dataset
data = load_dataset("json", data_files=args.data_path)

# Split dataset for training and validation
if args.val_set_size > 0:
    data = (
        data["train"]
        .train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
        .map(data_handler.generate_and_tokenize_prompt)
    )
else:
    data = data.shuffle(seed=42).map(data_handler.generate_and_tokenize_prompt)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# Load model based on finetuning method
if finetuning_method == "qlora":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    qlora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["k_proj", "v_proj", "q_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, qlora_config)
elif finetuning_method == "lora":
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["k_proj", "v_proj", "q_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
elif finetuning_method == "galore":
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
else:  # Supervised Fine-Tuning (SFT)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

model.config.use_cache = False

# Define training arguments based on finetuning method
if finetuning_method == "qlora":
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        bf16=True,
        logging_dir=logging_dir,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        report_to="wandb",
        optim="paged_adamw_8bit",
    )
elif finetuning_method == "galore":
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        bf16=True,
        logging_dir=logging_dir,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        report_to="wandb",
        optim="galore_adamw",
        optim_target_modules=["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
else:
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        bf16=True,
        logging_dir=logging_dir,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        report_to="wandb",
    )

# Utility to print trainable parameters
def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")

# Print trainable parameters before training
print_trainable_parameters(model)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"] if args.val_set_size > 0 else None,
    data_collator=data_collator,
)

if finetuning_method in ["sft", "lora"]:
    # Use Torch AdamW for SFT and LoRA
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = (
        args.epochs * len(data["train"]) // (args.batch_size * training_args.gradient_accumulation_steps)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    trainer.optimizers = (optimizer, scheduler)  # Manually set optimizers

# Train model
trainer.train()

# Save the final model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete. Model and tokenizer saved at:", output_dir)
wandb.finish()