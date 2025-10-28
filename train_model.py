import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SKLEARN_NO_OPENMP", "1")

import pickle
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATASET_PATH = "./dataset/heart_attack_prediction_indonesia.csv"
ARTIFACT_DIR = "artifacts"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Replica a etapa de limpeza do notebook."""
    df_clean = df.copy()
    df_clean["alcohol_missing"] = df_clean["alcohol_consumption"].isnull().astype(int)

    if df_clean["alcohol_consumption"].isnull().sum() > 0:
        mode_value = df_clean["alcohol_consumption"].mode()[0]
        df_clean["alcohol_consumption"] = df_clean["alcohol_consumption"].fillna(mode_value)

    df_clean = df_clean.drop_duplicates()
    return df_clean


def encode_categoricals(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Aplica LabelEncoder em cada coluna categórica, assim como no notebook."""
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X_encoded = X.copy()
    label_encoders: Dict[str, LabelEncoder] = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        X_encoded[col] = encoder.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = encoder

    return X_encoded, label_encoders


def train_and_select_model(X_train, y_train, X_val, y_val):
    """Treina os modelos avaliados no notebook e retorna o melhor pelo F1-Score."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, solver="liblinear"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=1),
        "KNN": KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree", n_jobs=1),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_proba) if y_val_proba is not None else None

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
            }
        )

        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    if results_df.empty:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("\nRESULTADOS NA VALIDAÇÃO (ordenados por F1-Score):")
    print(results_df.to_string(index=False))
    print(f"\nModelo selecionado: {best_model_name}")

    return best_model_name, best_model, results_df


def main():
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")

    df_clean = clean_dataset(df)
    print(f"Dataset após limpeza: {df_clean.shape[0]} linhas e {df_clean.shape[1]} colunas.")

    X = df_clean.drop("heart_attack", axis=1)
    y = df_clean["heart_attack"]
    feature_names = X.columns.tolist()

    X_encoded, label_encoders = encode_categoricals(X)

    # Divisão 70/15/15 com estratificação, replicando notebook.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(
        "Divisão dos dados:\n"
        f"  Treino: {X_train.shape[0]} samples\n"
        f"  Validação: {X_val.shape[0]} samples\n"
        f"  Teste: {X_test.shape[0]} samples"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    best_model_name, best_model, _ = train_and_select_model(
        X_train_scaled, y_train, X_val_scaled, y_val
    )

    # Avaliação no conjunto de teste.
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None

    print("\nAVALIAÇÃO NO CONJUNTO DE TESTE:")
    print(f"  Modelo: {best_model_name}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    if test_roc_auc is not None:
        print(f"  ROC-AUC: {test_roc_auc:.4f}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    with open(os.path.join(ARTIFACT_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(ARTIFACT_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)

    with open(os.path.join(ARTIFACT_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print("\nArtefatos salvos em 'artifacts/'.")


if __name__ == "__main__":
    main()
