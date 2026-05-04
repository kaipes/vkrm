import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import json


def train_isolation_forest_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    test_labels: np.ndarray,
    scaler: StandardScaler,
    model_path: str = "isolation_forest_model.pkl",
    metrics_path: str = "isolation_forest_metrics.json"
) -> Tuple[IsolationForest, dict]:
    """
    Обучает модель Isolation Forest для обнаружения аномалий.

    Args:
        train_data (pd.DataFrame): Обучающие данные (без меток).
        test_data (pd.DataFrame): Тестовые данные (без меток).
        test_labels (np.ndarray): Метки для теста (1 — аномалия, 0 — норма).
        scaler (StandardScaler): Масштабировщик, обученный на тренировке.
        model_path (str): Путь для сохранения модели.
        metrics_path (str): Путь для сохранения метрик.

    Returns:
        Tuple: (обученная модель, словарь с метриками)
    """

    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data["label"] = 0
    test_data["label"] = test_labels

    full_data = pd.concat([train_data, test_data], ignore_index=True)
    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = full_data.drop(columns=["label"])
    y = full_data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    contamination = y_train.mean()
    print(f"Доля аномалий в обучающей выборке: {contamination:.4f}")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    model.fit(X_train_scaled)

    y_pred = model.predict(X_test_scaled)
    y_pred = np.where(y_pred == -1, 1, 0)  # -1 → 1 (аномалия), 1 → 0 (норма)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred)
    }

    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")

    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Метрики сохранены в {metrics_path}")

    return model, metrics
