
import os, json, inspect
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

SYSTEM_PROMPT = (
    "You are a Lean 4 proof completion model.\n"
    "Given a Lean 4 sketch that contains exactly ONE <proof> hole, you must output ONLY the Lean code "
    "that should replace the <proof>.\n"
    "Ignore any 'sorry' in the input; do not try to replace or modify it.\n"
    "Do not output explanations. Do not repeat the input.\n"
    "Do NOT include the token 'sorry' anywhere in your output.\n"
    "End your output with the token <eofill>.\n"
)

def build_messages(header: str, theorem: str, sketch_state: str, target: str):
    user = (
        "Replace the <proof> in the following Lean 4 sketch.\n"
        "Ignore any occurrences of 'sorry'.\n\n"
        "Do NOT include the token 'sorry' anywhere in your output.\n"
        "[HEADER]\n"
        f"{header}\n\n"
        "[THEOREM+SKETCH]\n"
        f"{theorem}\n"
        f"{sketch_state}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]

def main():
    cfg_path = os.environ.get("RUN_CONFIG_PATH", "")
    assert cfg_path and Path(cfg_path).exists(), f"RUN_CONFIG_PATH not found: {cfg_path}"
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

    base_model = cfg["base_model"]
    out_dir = cfg["out_dir"]
    train_jsonl = cfg["stepwise_train"]
    eval_jsonl  = cfg["stepwise_valid"]
    

    max_len = int(cfg["max_len"])
    epochs  = float(cfg["epochs"])
    lr      = float(cfg["lr"])
    wd      = float(cfg["weight_decay"])

    bs      = int(cfg["batch_per_device"])
    gas     = int(cfg["grad_acc"])
    eval_steps = int(cfg["eval_steps"])
    save_steps = int(cfg["save_steps"])
    logging_steps = int(cfg["logging_steps"])

    lora_r = int(cfg["lora_r"])
    lora_alpha = int(cfg["lora_alpha"])
    lora_dropout = float(cfg["lora_dropout"])

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<proof>", "<eofill>"]})
    tokenizer.model_max_length = max_len  # 兜底截断

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    ds = load_dataset("json", data_files={"train": train_jsonl, "eval": eval_jsonl})
    
    ds["train"] = ds["train"].shuffle(seed=cfg["seed"])

    # ✅ 关键：trl 0.26.2 这里传进来的是 “单条样本 dict”，不是 batched
    def formatting_func(example: dict) -> str:
        header = example.get("header", "") or ""
        theorem = example.get("theorem", "") or ""
        sketch_state = example.get("sketch_state", "") or ""
        target = example.get("target", "") or ""

        msgs = build_messages(header, theorem, sketch_state, target)

        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        else:
            return (
                f"<<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
                f"<<USER>>\n{msgs[1]['content']}\n<</USER>>\n\n"
                f"<<ASSISTANT>>\n{msgs[2]['content']}\n<</ASSISTANT>>\n"
            )

    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=gas,
        learning_rate=lr,
        weight_decay=wd,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    sft_allowed = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        peft_config=peft_cfg,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    # 如果某些版本支持这些参数就加（不支持就不加）
    if "packing" in sft_allowed:
        trainer_kwargs["packing"] = False
    if "max_seq_length" in sft_allowed:
        trainer_kwargs["max_seq_length"] = max_len
    elif "max_length" in sft_allowed:
        trainer_kwargs["max_length"] = max_len

    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in sft_allowed}
    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    
    if trainer.is_world_process_zero():
        Path(out_dir, "log_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
        )

    
    metrics_eval = trainer.evaluate()


    if trainer.is_world_process_zero():
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        Path(out_dir, "metrics_eval.json").write_text(json.dumps(metrics_eval, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
