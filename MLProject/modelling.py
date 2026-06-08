import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Library untuk Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Library untuk MLOps
import mlflow
import mlflow.sklearn


def setup_mlflow_dagshub():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dagshub_file = os.path.join(base_dir, "DagsHub.txt")

    if not os.path.exists(dagshub_file):
        raise FileNotFoundError("File DagsHub.txt tidak ditemukan!")

    with open(dagshub_file, "r") as f:
        mlflow_url = f.read().strip()

    # Set alamat remote MLflow Cloud (Bersih tanpa modifikasi URL)
    mlflow.set_tracking_uri(mlflow_url)

    if "GITHUB_ACTIONS" in os.environ:
        print("[INFO] Environment: GitHub Actions. Menggunakan variable env standar.")
    else:
        import dagshub

        parts = mlflow_url.split("/")
        dagshub.init(repo_owner=parts[3], repo_name=parts[4].split(".")[0], mlflow=True)


def main():
    setup_mlflow_dagshub()
    mlflow.set_experiment("Predictive_Maintenance_Experiment")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir, "namadataset_preprocessing/predictive_maintenance_clean.csv"
    )

    print(f"[1/5] Memuat data bersih dari: {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Target"])
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    run = mlflow.active_run()
    is_project_run = run is not None

    if not is_project_run:
        run = mlflow.start_run(run_name="Random_Forest_Manual_Log")

    try:
        print("[2/5] Melatih model Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Manual logging ke DagsHub Cloud
        print("[3/5] Mengirim parameter dan metrik ke DagsHub...")
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print("[4/5] Membuat grafik artefak...")
        os.makedirs("temp_artifacts", exist_ok=True)
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Rusak"],
            yticklabels=["Normal", "Rusak"],
        )
        plt.title("Confusion Matrix")
        cm_path = "temp_artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # 5. Unggah model ke DagsHub Cloud (Syarat Kriteria 2 & 3)
        print("[5/5] Mengunggah model ke MLflow Cloud Artifacts...")
        mlflow.sklearn.log_model(model, "predictive_maintenance_model")

        # KUNCI SUKSES: Simpan salinan model secara lokal untuk kebutuhan Docker Build (Anti Gagal)
        local_model_path = os.path.join(base_dir, "predictive_maintenance_model_local")
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)

        mlflow.sklearn.save_model(model, local_model_path)
        print(f"[INFO] Salinan model lokal berhasil diamankan di: {local_model_path}")

    finally:
        if not is_project_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
