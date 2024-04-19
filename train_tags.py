import torch

torch.set_float32_matmul_precision("medium")

from fire import Fire
from transformers import (
    TrainingArguments,
    set_seed,
)
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd
from datasets import Dataset


def main(df_path: str = "dataset_tags.csv"):
    set_seed(42)

    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    max_seq_length = 8000
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    df = pd.read_csv(df_path)
    train_df = df[df["is_train_by_cluster"]].reset_index(drop=True)[
        ["original_text", "rewritten_text", "rewrite_prompt_style"]
    ]
    valid_df = df[~df["is_train_by_cluster"]].reset_index(drop=True)[
        ["original_text", "rewritten_text", "rewrite_prompt_style"]
    ]
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    def token_len(text):
        tokenized = tokenizer(text, return_length=True)
        length = tokenized["length"][0]
        return length

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["rewritten_text"])):
            ori_text = example["original_text"][i]
            rew_text = example["rewritten_text"][i]
            rew_prompt = example["rewrite_prompt_style"][i].rstrip(".").rstrip(":").strip()
            text = (
                "[INST]Given the original and rewritten text, provide keywords or tags that highlight the differences between the two texts."
                f'\n\nOriginal text: \n"""{ori_text}"""\n\n\nRewritten text: \n"""{rew_text}"""\n\n[/INST] ###Tags: \n"""{rew_prompt}"""</s>'
            )
            if token_len(text) > max_seq_length:
                continue
            output_texts.append(text)
        return output_texts

    response_template = '###Tags: \n"""'
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir="./output",
        do_train=True,
        do_eval=True,
        remove_unused_columns=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="steps",
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        weight_decay=1e-2,
        max_steps=175,
        eval_steps=20,
        save_steps=20,
        logging_steps=20,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=8,
        max_grad_norm=1.0,
        gradient_accumulation_steps=64,
        overwrite_output_dir=False,
        eval_accumulation_steps=1,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        compute_metrics=None,
        max_seq_length=max_seq_length,
        formatting_func=formatting_prompts_func,
    )
    trainer.train()


if __name__ == "__main__":
    Fire(main)
