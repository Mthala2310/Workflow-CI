import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Library untuk MLOps (DagsHub & MLflow)
import mlflow
import mlflow.sklearn
import dagshub


def setup_mlflow_dagshub():
    """Fungsi untuk menghubungkan MLflow lokal ke server Cloud DagsHub."""
    # Membaca tautan dari DagsHub.txt
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dagshub_file = os.path.join(base_dir, "DagsHub.txt")

    if not os.path.exists(dagshub_file):
        raise FileNotFoundError("File DagsHub.txt tidak ditemukan!")

    with open(dagshub_file, "r") as f:
        mlflow_url = f.read().strip()

    # Konfigurasi MLflow Remote URL
    mlflow.set_tracking_uri(mlflow_url)

    # Ekstrak username dan repo name dari URL untuk inisialisasi DagsHub
    # URL format: https://dagshub.com/username/repo_name.mlflow
    parts = mlflow_url.split("/")
    username = parts[3]
    repo_name = parts[4].split(".")[0]

    print(f"[INFO] Menginisialisasi DagsHub untuk {username}/{repo_name}")
    dagshub.init(repo_owner=username, repo_name=repo_name, mlflow=True)


def main():
    # 1. Setup Koneksi ke DagsHub
    setup_mlflow_dagshub()

    # Set nama eksperimen di MLflow
    mlflow.set_experiment("Predictive_Maintenance_Experiment")

    # 2. Memuat Dataset Hasil Preprocessing Kriteria 1
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir, "namadataset_preprocessing/predictive_maintenance_clean.csv"
    )

    print(f"[1/5] Memuat data bersih dari: {data_path}")
    df = pd.read_csv(data_path)

    # Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Split data menjadi Train set (80%) dan Test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Menentukan Hyperparameter Model
    n_estimators = 100
    max_depth = 10
    class_weight = "balanced"  # Strategi ampuh untuk mengatasi data jomplang!

    # 4. Memulai Pencatatan Eksperimen Manual di MLflow (Wajib untuk Advance)
    with mlflow.start_run(run_name="Random_Forest_Manual_Log"):
        print("[2/5] Melatih model Random Forest...")

        # Inisialisasi dan latih model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Prediksi ke data test
        y_pred = model.predict(X_test)

        # 5. Menghitung Metrik Evaluasi
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(
            f"      Metrik -> Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
        )

        # 6. MANUAL LOGGING (Pencatatan Manual ke MLflow Cloud)
        print("[3/5] Mengirim parameter dan metrik ke DagsHub...")
        # Log Hyperparameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("class_weight", class_weight)

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # 7. MEMBUAT DAN MENYIMPAN 2 ARTEFAK TAMBAHAN (Syarat Mutlak Advance)
        print("[4/5] Membuat grafik artefak tambahan...")
        os.makedirs("temp_artifacts", exist_ok=True)

        # Artefak 1: Grafik Confusion Matrix
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
        plt.title("Confusion Matrix - Predictive Maintenance")
        plt.ylabel("Aktual")
        plt.xlabel("Prediksi")
        cm_path = "temp_artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # Artefak 2: Grafik Feature Importance (Fitur paling berpengaruh)
        plt.figure(figsize=(8, 5))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
        plt.title("Feature Importance")
        plt.xlabel("Skor Kepentingan")
        fi_path = "temp_artifacts/feature_importance.png"
        plt.savefig(fi_path)
        plt.close()

        # Unggah artefak tersebut ke MLflow Cloud DagsHub
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)

        # 8. Menyimpan Model Utama ke MLflow Artifacts
        print("[5/5] Mengunggah model ke MLflow Artifacts...")
        mlflow.sklearn.log_model(model, "predictive_maintenance_model")

        print("\n=== Eksperimen Selesai! Silakan cek Dashboard DagsHub Anda ===")


if __name__ == "__main__":
    main()
