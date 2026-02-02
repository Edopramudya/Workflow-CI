import numpy as np
import mlflow
import mlflow.sklearn
import os
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# BASE DIR
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LOAD DATA
# =========================
X = np.load(os.path.join(BASE_DIR, "X.npy"))
y = np.load(os.path.join(BASE_DIR, "y.npy"))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("Titanic_Basic_Experiment")

# =========================
# MLFLOW RUN
# =========================
with mlflow.start_run():

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log metric
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()

    cm_path = os.path.join(BASE_DIR, "training_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # =========================
    # METRIC INFO (JSON)
    # =========================
    metric_info = {
        "accuracy": acc,
        "model_type": "LogisticRegression",
        "dataset": "Titanic",
        "n_features": X.shape[1],
        "test_size": 0.2
    }

    json_path = os.path.join(BASE_DIR, "metric_info.json")
    with open(json_path, "w") as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact(json_path)

    # =========================
    # ESTIMATOR INFO (HTML)
    # =========================
    html_path = os.path.join(BASE_DIR, "estimator.html")
    with open(html_path, "w") as f:
        f.write(str(model))

    mlflow.log_artifact(html_path)

    print("Accuracy:", acc)
