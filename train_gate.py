import torch

torch.set_float32_matmul_precision("medium")

from transformers import AutoTokenizer
from fire import Fire
from transformers import (
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers.models.mistral.modeling_mistral import (
    MistralForSequenceClassification,
    MistralConfig,
)

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast


class PromptDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data: pd.DataFrame,
        transforms=None,
        mode: str = "train",
        ignore_id: int = -100,
        max_length: int = 2048,
    ):
        self.data = data
        self.transforms = transforms
        self.mode = mode
        self.tokenizer = tokenizer
        self.ignore_id = ignore_id
        self.max_length = max_length

    def __getitem__(self, idx: int) -> tuple:
        row = self.data.iloc[idx]
        original_text = row["original_text"]
        rewrite_prompt = row["rewrite_prompt"]
        rewritten_text = row["rewritten_text"]

        if np.random.random() < 0.4:
            labels = torch.tensor(True).float().view(1)
        elif np.random.random() < 0.6:
            labels = torch.tensor(False).float().view(1)
            pool_idx = set(range(len(self.data))) - {idx}
            idx = np.random.choice(list(pool_idx))
            rewrite_prompt = self.data.iloc[idx]["rewrite_prompt"]
        else:
            # hard samples
            labels = torch.tensor(False).float().view(1)
            pool_idx = [
                x
                for x in ast.literal_eval(row["ohem_perma_ids"])
                if x in self.data["perma_idx"].values
            ]
            if len(pool_idx):
                idx = np.random.choice(list(pool_idx))
                rewrite_prompt = self.data[self.data["perma_idx"] == idx]["rewrite_prompt"].iloc[0]
            else:
                # any example
                pool_idx = set(range(len(self.data))) - {idx}
                idx = np.random.choice(list(pool_idx))
                rewrite_prompt = self.data.iloc[idx]["rewrite_prompt"]

        prompt = (
            "<s>[INST]Given the original text and a candidate prompt for rewriting it, you are expected to evaluate if the rewritten text makes sense given the candidate prompot. You will have positive examples where the rewritten text really was created by applied the proposed prompt and negative samples where the rewritten text was created by a different prompt."
            f'\n\nOriginal text: \n"""{original_text}"""\n\nCandidate Prompt: \n"""{rewrite_prompt}"""\n\nRewritten text: \n"""{rewritten_text}"""[/INST]'
        )

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.data)


def setup_datasets(tokenizer, df_path: str, fold: int):
    df_data = pd.read_csv(df_path)

    train_dataset = PromptDataset(
        tokenizer,
        df_data[df_data.fold != fold].copy().reset_index(drop=True),
        mode="train",
    )
    print(f"training dataset: {len(train_dataset)}")

    valid_dataset = PromptDataset(
        tokenizer,
        df_data[df_data.fold == fold].copy().reset_index(drop=True),
        mode="valid",
    )
    print(f"validation dataset: {len(valid_dataset)}")
    return train_dataset, valid_dataset


def main(df_path: str = "dataset.csv", fold: int = 0):
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token

    train_dataset, valid_dataset = setup_datasets(tokenizer, df_path, fold)
    collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer, padding="longest")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    mistral_cfg = MistralConfig.from_pretrained(model_name)
    mistral_cfg.num_labels = 1
    mistral_cfg.problem_type = "multi_label_classification"
    model: PreTrainedModel = MistralForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        config=mistral_cfg,
    )
    model.score.weight.data = model.score.weight.data.to(torch.float32)

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        # task_type=TaskType.CAUSAL_LM,
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=128,
        lora_alpha=128,
        use_rslora=False,
        lora_dropout=0.05,
        bias="none",  # Supports any, but = "none" is optimized
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./output",
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="steps",
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True,
        optim="paged_adamw_8bit",
        weight_decay=1e-2,
        max_steps=250,
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=8,
        max_grad_norm=1.0,
        gradient_accumulation_steps=128,
        overwrite_output_dir=False,
        eval_accumulation_steps=1,
        seed=42,
    )

    trainer = Trainer(  # type: ignore
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        compute_metrics=None,
    )
    trainer.train()


if __name__ == "__main__":
    Fire(main)
