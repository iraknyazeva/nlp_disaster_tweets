import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
from datasets import Dataset

from src.utils.seed import set_seed
from src.preprocessing.clean import apply_preprocessing
from src.features.embeddings import get_embeddings

# загружаем валидационную выборку
def load_val_data(data_cfg):
    train = pd.read_csv(data_cfg["train_path"])
    texts = train[data_cfg["text_column"]].tolist()
    y = train[data_cfg["target_column"]]

    _, X_val_text, _, y_val = train_test_split(
        texts, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["seed"],
        stratify=y
    )
    return X_val_text, y_val

# получаем эмбеддинги от word2vec
def get_w2v_embeddings(texts, w2v):
    embeddings = []
    for text in texts:
        tokens = text.split()
        vectors = [w2v.wv[token] for token in tokens if token in w2v.wv]
        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(w2v.vector_size))
    return np.array(embeddings)

# вспомогательные функции для загрузки моделей и получения предсказаний
def eval_tfidf_logreg(model_path, texts, y_true, preprocess):
    model = joblib.load(model_path)
    processed = apply_preprocessing(texts, preprocess)
    preds = model.predict(processed)
    return f1_score(y_true, preds), accuracy_score(y_true, preds)

def eval_tfidf_catboost(model_path, texts, y_true, preprocess):
    model = joblib.load(model_path)
    vec_path = model_path.replace(".pkl", "_vectorizer.pkl")
    vectorizer = joblib.load(vec_path)
    processed = apply_preprocessing(texts, preprocess)
    X = vectorizer.transform(processed)
    preds = model.predict(X)
    return f1_score(y_true, preds), accuracy_score(y_true, preds)

def eval_word2vec(model_path, texts, y_true, preprocess):
    model = joblib.load(model_path)
    processed = apply_preprocessing(texts, preprocess)

    # logreg (тк теперь сохраняется pipeline)
    if hasattr(model, "named_steps"):
        preds = model.predict(processed)
        return f1_score(y_true, preds), accuracy_score(y_true, preds)

    # catboost (сохраняю отдельно модель и w2v)
    w2v = joblib.load(model_path.replace(".pkl", "_vectorizer.pkl"))
    X = get_w2v_embeddings(processed, w2v)

    preds = model.predict(X)
    return f1_score(y_true, preds), accuracy_score(y_true, preds)

def eval_minilm(model_path, texts, y_true, batch_size, preprocess=None):
    if preprocess and preprocess != "raw":
      texts = apply_preprocessing(texts ,preprocess)

    model = joblib.load(model_path)
    embeddings = get_embeddings(
        texts,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=batch_size,
        show_progress_bar=False
    )
    preds = model.predict(embeddings)
    return f1_score(y_true, preds), accuracy_score(y_true, preds)

def eval_distilbert(model_path, texts, y_true):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128
        )

    dataset = Dataset.from_dict({"text": texts, "label": y_true})
    dataset = dataset.map(tokenize, batched=True)

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    predictions = trainer.predict(dataset).predictions
    preds = np.argmax(predictions, axis=-1)

    return f1_score(y_true, preds), accuracy_score(y_true, preds)

def main():
    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    with open("configs/embeddings.yaml") as f:
        emb_cfg = yaml.safe_load(f)

    batch_size = emb_cfg.get("batch_size", 32)

    set_seed(data_cfg["seed"])
    X_val_text, y_val = load_val_data(data_cfg)

    results = []

    # tf-idf + logreg
    for prep in ["v1", "v2", "v3"]:
        path = f"models/tfidf_logreg_{prep}.pkl"
        if Path(path).exists():
            f1, acc = eval_tfidf_logreg(path, X_val_text, y_val, prep)
            results.append({"model": "tfidf_logreg", "preprocess": prep, "f1": round(f1, 4), "accuracy": round(acc, 4)})

    # tf-idf + catboost
    for prep in ["v1", "v2", "v3"]:
        path = f"models/tfidf_catboost_{prep}.pkl"
        if Path(path).exists():
            f1, acc = eval_tfidf_catboost(path, X_val_text, y_val, prep)
            results.append({"model": "tfidf_catboost", "preprocess": prep, "f1": round(f1, 4), "accuracy": round(acc, 4)})

    # word2vec + logreg
    for prep in ["v1", "v2", "v3"]:
        path = f"models/word2vec_logreg_{prep}.pkl"
        if Path(path).exists():
            f1, acc = eval_word2vec(path, X_val_text, y_val, prep)
            results.append({"model": "word2vec_logreg", "preprocess": prep, "f1": round(f1, 4), "accuracy": round(acc, 4)})

    # word2vec + catboost
    for prep in ["v1", "v2", "v3"]:
        path = f"models/word2vec_catboost_{prep}.pkl"
        if Path(path).exists():
            f1, acc = eval_word2vec(path, X_val_text, y_val, prep)
            results.append({"model": "word2vec_catboost", "preprocess": prep, "f1": round(f1, 4), "accuracy": round(acc, 4)})

    # miniLM
    for suffix, prep_name in [("raw", "raw"), ("prep_v1", "v1")]:
        path = f"models/minilm_logreg_{suffix}.pkl"
        if Path(path).exists():
            f1, acc = eval_minilm(path, X_val_text, y_val, batch_size, prep_name)
            results.append({"model": "minilm_logreg", "preprocess": prep_name, "f1": round(f1, 4), "accuracy": round(acc, 4)})

    # distilBert
    path = "models/distilbert_finetuned"
    if Path(path).exists():
        f1, acc = eval_distilbert(path, X_val_text, y_val)
        results.append({"model": "distilbert", "preprocess": "raw", "f1": round(f1, 4), "accuracy": round(acc, 4)})

    # результаты отсортированные по f1
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("f1", ascending=False)
        print("\n" + "="*60)
        print(df.to_string(index=False))
        print("="*60)
    else:
        print("нет моделей в папке models/")

if __name__ == "__main__":
    main()