import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import yaml
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

from src.utils.seed import set_seed
from src.preprocessing.clean import apply_preprocessing

@click.command()
@click.option("--config", default="configs/classic_ml.yaml", help="Path to config")
@click.option("--mode", type=click.Choice(["tfidf", "word2vec"]), default="tfidf")
@click.option("--preprocess", type=click.Choice(["v1", "v2", "v3"]), default="v1")
@click.option("--model", type=click.Choice(["logreg", "catboost"]), default="logreg")
def main(config, mode, preprocess, model):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    train = pd.read_csv(data_cfg["train_path"])
    texts = apply_preprocessing(train[data_cfg["text_column"]], preprocess)
    y = train[data_cfg["target_column"]]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, y,
        test_size=data_cfg["test_size"],
        random_state=cfg["seed"],
        stratify=y if data_cfg.get("stratify", True) else None
    )

    Path("models").mkdir(exist_ok=True)
    model_name = f"{mode}_{model}_{preprocess}"

    if mode == "tfidf":
        if model == "logreg":
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
                    max_features=cfg["tfidf"]["max_features"],
                    min_df=cfg["tfidf"]["min_df"]
                )),
                ("clf", LogisticRegression(
                    class_weight=cfg["logreg"]["class_weight"],
                    max_iter=cfg["logreg"]["max_iter"],
                    random_state=cfg["seed"]
                ))
            ])
            param_grid = {f"clf__{k}": v for k, v in cfg["logreg"]["param_grid"].items()}

            grid = GridSearchCV(
                pipe, param_grid,
                cv=cfg["cv_folds"],
                scoring=cfg["scoring"],
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            print(f"CV F1: {grid.best_score_:.4f}")

        elif model == "catboost":
            vectorizer = TfidfVectorizer(
                ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
                max_features=cfg["tfidf"]["max_features"],
                min_df=cfg["tfidf"]["min_df"]
            )
            X_train_vec = vectorizer.fit_transform(X_train)
            X_val_vec = vectorizer.transform(X_val)

            cb = CatBoostClassifier(
                iterations=cfg["catboost"]["iterations"],
                learning_rate=cfg["catboost"]["learning_rate"],
                depth=cfg["catboost"]["depth"],
                verbose=cfg["catboost"]["verbose"],
                early_stopping_rounds=cfg["catboost"]["early_stopping_rounds"],
                random_seed=cfg["seed"]
            )
            cb.fit(X_train_vec, y_train, eval_set=(X_val_vec, y_val))
            best_model = cb
            joblib.dump(vectorizer, f"models/{model_name}_vectorizer.pkl")

        preds = best_model.predict(X_val_vec if model == "catboost" else X_val)
        f1 = f1_score(y_val, preds)
        accuracy = accuracy_score(y_val, preds)
        print(f"Val f1: {f1:.4f}")
        print(f"Val accuracy: {f1:.4f}")
        joblib.dump(best_model, f"models/{model_name}.pkl")

    else:
        click.echo("-")

if __name__ == "__main__":
    main()