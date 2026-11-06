import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import yaml
import pandas as pd
import numpy as np
import optuna
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

from src.utils.seed import set_seed
from src.preprocessing.clean import apply_preprocessing

def get_w2v_texts(texts, w2v):
    embeddings = []
    for text in texts:
        tokens = text.split()
        vectors = [w2v.wv[token] for token in tokens if token in w2v.wv]
        if len(vectors) > 0:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(w2v.vector_size))
    return np.array(embeddings)

def train_catboost_with_optuna(X_train, y_train, X_val, y_val, cfg, n_trials=20):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_seed": cfg["seed"],
            "verbose": 0,
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"])
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"Best params: {study.best_params}")
    print(f"Best F1 during search: {study.best_value:.4f}")

    best_params = study.best_params.copy()
    best_params.update({
        "random_seed": cfg["seed"],
        "verbose": 100,
        "early_stopping_rounds": 50,
    })
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return best_model

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
            preds = best_model.predict(X_val)

        elif model == "catboost":
            vectorizer = TfidfVectorizer(
                ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
                max_features=cfg["tfidf"]["max_features"],
                min_df=cfg["tfidf"]["min_df"]
            )
            X_train_vec = vectorizer.fit_transform(X_train)
            X_val_vec = vectorizer.transform(X_val)

            best_model = train_catboost_with_optuna(
                X_train_vec, y_train, X_val_vec, y_val, cfg
            )
            preds = best_model.predict(X_val_vec)
            joblib.dump(vectorizer, f"models/{model_name}_vectorizer.pkl")

        f1 = f1_score(y_val, preds)
        accuracy = accuracy_score(y_val, preds)
        print(f"Val F1: {f1:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        joblib.dump(best_model, f"models/{model_name}.pkl")

    elif mode == "word2vec":
        tokenized = [text.split() for text in X_train]
        w2v = Word2Vec(
            sentences=tokenized,
            vector_size=300,
            window=3,
            negative=10,
            min_count=2,
            seed=cfg["seed"]
        )

        X_train_emb = get_w2v_texts(X_train, w2v)
        X_val_emb = get_w2v_texts(X_val, w2v)

        if model == "logreg":
            logreg = LogisticRegression(
                class_weight=cfg["logreg"]["class_weight"],
                max_iter=cfg["logreg"]["max_iter"],
                random_state=cfg["seed"]
            )
            param_grid = cfg["logreg"]["param_grid"]
            grid = GridSearchCV(
                logreg, param_grid,
                cv=cfg["cv_folds"],
                scoring=cfg["scoring"],
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X_train_emb, y_train)
            best_model = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            print(f"CV F1: {grid.best_score_:.4f}")
            preds = best_model.predict(X_val_emb)

        elif model == "catboost":
            best_model = train_catboost_with_optuna(
                X_train_emb, y_train, X_val_emb, y_val, cfg
            )
            preds = best_model.predict(X_val_emb)

        f1 = f1_score(y_val, preds)
        accuracy = accuracy_score(y_val, preds)
        print(f"Val F1: {f1:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        joblib.dump(best_model, f"models/{model_name}.pkl")
        joblib.dump(w2v, f"models/{model_name}_w2v.pkl")

if __name__ == "__main__":
    main()