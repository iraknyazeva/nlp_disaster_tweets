import sys
from pathlib import Path

# добавляем корень проекта в sys.path для импортов из src 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import yaml
import pandas as pd
import numpy as np
import optuna
import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from src.utils.seed import set_seed
from src.preprocessing.clean import apply_preprocessing
from src.features.w2v_transformer import get_w2v_texts

from src.features.w2v_transformer import Word2VecTransformer

# подбор гиперпараметров catboost с помощью optuna 
def train_catboost_tfidf_with_optuna(X_train, y_train, cfg, vectorizer_params):
    search_space = cfg["catboost"]["search_space"]
    n_trials = cfg["catboost"].get("n_trials", 20)
    early_stopping = cfg["catboost"].get("early_stopping_rounds", 50)
    cv_folds = cfg.get("cv_folds", 5)

    def objective(trial):    
        params = {
            "random_seed": cfg["seed"],
            "verbose": 0,
            "early_stopping_rounds": early_stopping,
        }
        # если gpu доступно, обучаем на gpu 
        if torch.cuda.is_available():
            params["task_type"] = "GPU"

        for name, space in search_space.items():
            if space["type"] == "int":
                params[name] = trial.suggest_int(name, space["low"], space["high"])
            elif space["type"] == "float":
                params[name] = trial.suggest_float(
                    name,
                    space["low"],
                    space["high"],
                    log=space.get("log", False)
                )

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg["seed"])
        scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
          X_tr = [X_train[i] for i in train_idx]
          X_va = [X_train[i] for i in val_idx]
          y_tr = y_train.iloc[train_idx]
          y_va = y_train.iloc[val_idx]

          vectorizer = TfidfVectorizer(**vectorizer_params)
          X_tr = vectorizer.fit_transform(X_tr)
          X_va = vectorizer.transform(X_va)
          
          model = CatBoostClassifier(**params)
          model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            verbose=0
          )

          preds = model.predict(X_va)
          scores.append(f1_score(y_va, preds))

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"])
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"Best params: {study.best_params}")
    print(f"CV F1: {study.best_value:.4f}")

    # обучаем финальную модель с лучшими параметрами 
    best_params = study.best_params.copy()
    best_params.update({
        "random_seed": cfg["seed"],
        "verbose": 100,
        "early_stopping_rounds": early_stopping,
    })

    if torch.cuda.is_available():
        best_params["task_type"] = "GPU"

    # после подбора гиперпараметров vectorizer уже на всем train 
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train = vectorizer.fit_transform(X_train)

    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # возвращает обученную модель 
    return best_model, vectorizer




@click.command()
@click.option("--config", default="configs/classic_ml.yaml", help="Path to config")
@click.option("--mode", type=click.Choice(["tfidf", "word2vec"]), default="tfidf")
@click.option("--preprocess", type=click.Choice(["v1", "v2", "v3"]), default="v1")
@click.option("--model", type=click.Choice(["logreg", "catboost"]), default="logreg")
def main(config, mode, preprocess, model):
    # загрузка конфигов
    with open(config) as f:
        cfg = yaml.safe_load(f)

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    # загружаем данные и применяем выбранный препроцесссинг 
    train = pd.read_csv(data_cfg["train_path"])
    texts = apply_preprocessing(train[data_cfg["text_column"]], preprocess)
    y = train[data_cfg["target_column"]]

    # разделение выборки на train/val
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
            # tfidf + logreg 
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
                verbose=2
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            print(f"CV F1: {grid.best_score_:.4f}")
            preds = best_model.predict(X_val)

        elif model == "catboost":
            vectorizer_params={
                "ngram_range": tuple(cfg["tfidf"]["ngram_range"]),
                "max_features": cfg["tfidf"]["max_features"],
                "min_df": cfg["tfidf"]["min_df"]
            }
            best_model, vectorizer = train_catboost_tfidf_with_optuna(X_train, y_train, cfg, vectorizer_params)
            X_val_vec = vectorizer.transform(X_val)
            preds = best_model.predict(X_val_vec)
            joblib.dump(vectorizer, f"models/{model_name}_vectorizer.pkl")

        # считаем метрики и сохраняем модель
        f1 = f1_score(y_val, preds)
        accuracy = accuracy_score(y_val, preds)
        print(f"Val F1: {f1:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        joblib.dump(best_model, f"models/{model_name}.pkl")

    elif mode == "word2vec":
        if model == "logreg":
            # word2vec + logreg
            # для logreg делаем стандартизацию эмбеддингов word2vec 
            pipe = Pipeline([
              ("w2v", Word2VecTransformer(
                  vector_size=cfg["word2vec"]["vector_size"],
                  window=cfg["word2vec"]["window"],
                  negative=cfg["word2vec"]["negative"],
                  min_count=cfg["word2vec"]["min_count"],
                  seed=cfg["seed"]
              )),
              ("scaler", StandardScaler()),
              ("clf", LogisticRegression(
                class_weight=cfg["logreg"]["class_weight"],
                max_iter=cfg["logreg"]["max_iter"],
                random_state=cfg["seed"]
              ))
            ])

            param_grid = {
              f"clf__{k}": v for k, v in cfg["logreg"]["param_grid"].items()
            }

            grid = GridSearchCV(
              pipe, 
              param_grid,
              cv=cfg["cv_folds"],
              scoring=cfg["scoring"],
              n_jobs=1,
              verbose=2
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            print(f"Best params:{grid.best_params_}")
            print(f"CV f1:{grid.best_score_:.4f}")

            preds = best_model.predict(X_val)

        elif model == "catboost":
            # word2vec + catboost 
            # best_model = train_catboost_with_optuna(X_train_emb, y_train, cfg)
            # preds = best_model.predict(X_val_emb)
            print()

        # считаем метрики и сохраняем модель 
        f1 = f1_score(y_val, preds)
        accuracy = accuracy_score(y_val, preds)
        print(f"Val F1: {f1:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        joblib.dump(best_model, f"models/{model_name}.pkl")

if __name__ == "__main__":
    main()