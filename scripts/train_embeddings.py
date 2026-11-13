import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import click
import yaml
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score

from src.utils.seed import set_seed
from src.features.embeddings import get_embeddings
from src.preprocessing.clean import apply_preprocessing

@click.command()
@click.option("--config", default="configs/embeddings.yaml")
@click.option("--use-preprocessed", is_flag=True, help="использовать ли предобработанный текст")
def main(config, use_preprocessed):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    train = pd.read_csv(data_cfg["train_path"])
    texts = train[data_cfg["text_column"]].tolist()

    if use_preprocessed:
        texts = apply_preprocessing(texts, "v1")
        vers = "prep_v1"
    else:
        print("используется сырой текст")
        vers = "raw"

    y = train[data_cfg["target_column"]]

    embeddings = get_embeddings(
        texts,
        model_name=cfg["model_name"],
        batch_size=cfg["batch_size"],
        show_progress_bar=cfg["show_progress_bar"]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, y,
        test_size=data_cfg["test_size"],
        random_state=cfg["seed"],
        stratify=y if data_cfg.get("stratify", True) else None
    )

    logreg = LogisticRegression(
        class_weight=cfg["logreg"]["class_weight"],
        max_iter=cfg["logreg"]["max_iter"],
        random_state=cfg["seed"]
    )

    grid = GridSearchCV(
        logreg,
        cfg["logreg"]["param_grid"],
        cv=cfg["cv_folds"],
        scoring=cfg["scoring"],
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_val)

    f1 = f1_score(y_val, preds)
    acc = accuracy_score(y_val, preds)

    print(f"Best params: {grid.best_params_}")
    print(f"CV f1: {grid.best_score_:.4f}")
    print(f"Val f1: {f1:.4f}")
    print(f"Val accuracy: {acc:.4f}")

    Path("models").mkdir(exist_ok=True)
    model_path = f"models/minilm_logreg_{vers}.pkl"
    joblib.dump(best_model, model_path)

if __name__ == "__main__":
    main()
