import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from src.utils.seed import set_seed

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def main():
    with open("configs/finetune.yaml") as f:
        cfg = yaml.safe_load(f)

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    train_df = pd.read_csv(data_cfg["train_path"])
    texts = train_df[data_cfg["text_column"]].tolist()
    labels = train_df[data_cfg["target_column"]].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=data_cfg["test_size"],
        random_state=cfg["seed"],
        stratify=labels
    )

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["max_length"]
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=cfg["training"]["warmup_steps"],
        eval_strategy=cfg["training"]["eval_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        greater_is_better=cfg["training"]["greater_is_better"],
        logging_steps=cfg["training"]["logging_steps"],
        fp16=cfg["training"]["fp16"],
        report_to="none"
    )

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      data_collator=DataCollatorWithPadding(tokenizer),
      compute_metrics=compute_metrics,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["training"]["early_stopping_patience"])]
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"Val F1: {metrics['eval_f1']:.4f}")
    print(f"Val Accuracy: {metrics['eval_accuracy']:.4f}")

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    main()