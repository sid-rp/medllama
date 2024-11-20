import os
import sys
from typing import Tuple, Union

import fire
import torch
from datasets import load_dataset,Features,Value
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
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)

def main(
    model: str,  # e.g. "decapoda-research/llama-7b-hf"
    val_set_size: Union[int, float] = 0.1,
    prompt_template: str = "prompt/medalpaca.json",
    model_max_length: Union[int, str] = 256,  # Accept as int or str for safety
    train_on_inputs: bool = True,
    data_path: str = "medical_meadow_small.json",
    train_in_8bit: bool = True,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Tuple[str] = ("q_proj", "v_proj"),
    per_device_batch_size: Union[int, str] = 1,
    num_epochs: Union[int, str] = 1,
    learning_rate: Union[float, str] = 5e-6,
    global_batch_size: Union[int, str] = 1,
    output_dir: str = "./output",
    save_total_limit: Union[int, str] = 3,
    eval_steps: Union[int, str] = 200,
    device_map: str = "auto",
    group_by_length: bool = False,
    wandb_run_name: str = "test",
    use_wandb: bool = False,
    wandb_project: str = "medalpaca",
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    warmup_steps: Union[int, str] = 100,
    fsdp: str = "full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap: str = "LlamaDecoderLayer",
    **kwargs
):
    # Convert any string inputs to integers or floats as necessary
    model_max_length = int(model_max_length)
    per_device_batch_size = int(per_device_batch_size)
    num_epochs = int(num_epochs)
    learning_rate = float(learning_rate)
    global_batch_size = int(global_batch_size)
    save_total_limit = int(save_total_limit)
    eval_steps = int(eval_steps)
    warmup_steps = int(warmup_steps)

    # Proceed with the rest of the function
    os.environ["WANDB_API_KEY"] = "949bc5b85ff83ca46ec5d139d69a50146a19850a"
    os.environ["WANDB_MODE"] = "offline"
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    # adapt arguments
    model_name = model
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = global_batch_size // per_device_batch_size
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        if use_lora:
            fsdp, fsdp_transformer_layer_cls_to_wrap = "", None
    else:
        fsdp, fsdp_transformer_layer_cls_to_wrap = "", None

    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    # perform some checks
    if fp16 and bf16:
        raise ValueError("At most one of fp16 and bf16 can be True, but not both.")

    if train_in_8bit and not use_lora:
        raise ValueError("8bit training without LoRA is not supported")

    # Configure quantization for 8-bit training
    if train_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    # init model
    if "llama" in model_name:
        load_model = LlamaForCausalLM
    else:
        load_model = AutoModelForCausalLM

    # Initialize model on CPU to minimize initial GPU memory fragmentation
    model = load_model.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if any([use_lora, bf16]) else torch.float32,
        device_map=device_map  # Load on CPU first
    )


    if train_in_8bit:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # init tokenizer and tokenize function
    try:
        # Attempt to load the tokenizer with AutoTokenizer, which is generally compatible with most models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"An error occurred while loading the tokenizer: {e}")
    # Handle any additional fallback logic here if needed

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # load and tokenize data
    data_handler = DataHandler(
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        model_max_length=model_max_length,
        train_on_inputs=train_on_inputs,
    )
    features = Features({
    "instruction": Value("string"),
    "input": Value("string"),
    "output": Value("string")
    # Add other fields based on your JSON structure
    })
    data = load_dataset("json", data_files=data_path,features=features)

    if float(val_set_size) > 0:
        data = (
            data["train"]
            .train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            .map(data_handler.generate_and_tokenize_prompt)
        )
    else:
        data = data.shuffle(seed=42).map(data_handler.generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps" if float(val_set_size) > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if val_set_size > 0 else None,
        save_steps=eval_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to=[] if not use_wandb else ["wandb"],
        run_name=wandb_run_name if use_wandb else None,
        fsdp=fsdp,
        fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls_to_wrap,
        **kwargs
    )

    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"] if val_set_size > 0 else None,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    if use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(output_dir)
    model.config.to_json_file("config.json")
if __name__ == "__main__":
    fire.Fire(main)
    
    
    
 