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

# Library untuk MLOps
import mlflow
import mlflow.sklearn


def setup_mlflow_dagshub():
    """Fungsi mendeteksi environment dan mengatur tracking URI."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dagshub_file = os.path.join(base_dir, "DagsHub.txt")

    if not os.path.exists(dagshub_file):
        raise FileNotFoundError("File DagsHub.txt tidak ditemukan!")

    with open(dagshub_file, "r") as f:
        mlflow_url = f.read().strip()

    # Set alamat remote MLflow Cloud
    mlflow.set_tracking_uri(mlflow_url)

    if "GITHUB_ACTIONS" in os.environ:
        print("[INFO] Environment: GitHub Actions. Autentikasi token aktif.")
    else:
        import dagshub

        parts = mlflow_url.split("/")
        print(
            f"[INFO] Environment: Lokal. Menginisialisasi DagsHub untuk {parts[3]}/{parts[4].split('.')[0]}"
        )
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

    # KUNCI UTAMA: Periksa apakah sudah ada Run aktif dari mlflow run
    run = mlflow.active_run()
    is_project_run = run is not None

    if not is_project_run:
        # Jika dijalankan manual di lokal, baru buat run baru
        print("[INFO] Tidak ada run aktif. Memulai run baru secara manual.")
        run = mlflow.start_run(run_name="Random_Forest_Manual_Log")
    else:
        print(
            f"[INFO] Menemukan run aktif dari MLflow Project (ID: {run.info.run_id}). Menggunakan run ini."
        )

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

        print(
            f"      Metrik -> Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
        )

        # Manual logging ke DagsHub
        print("[3/5] Mengirim parameter dan metrik ke DagsHub...")
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Membuat Grafik Artefak
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

        # Simpan Model
        print("[5/5] Mengunggah model ke MLflow Artifacts...")
        mlflow.sklearn.log_model(model, "predictive_maintenance_model")

        # Simpan RUN_ID asli ke file teks di luar folder agar bisa dibaca oleh step Docker
        run_id = run.info.run_id
        with open(os.path.join(base_dir, "../run_id.txt"), "w") as f:
            f.write(run_id)

        print(f"\n=== Proses Berhasil! ID Latihan: {run_id} ===")

    finally:
        # Akhiri run HANYA jika ini bukan diatur oleh mlflow run (mencegah penutupan prematur)
        if not is_project_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
