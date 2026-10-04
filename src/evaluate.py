import logging
import os
import warnings

import mlflow

from loader import get_train_test_split_data

# --- Setup Logging ---
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# --- Configuration ---
DATA_PATH = "data/telco_churn.csv"
EXPERIMENT_NAME = "Churn_Prediction_Basic"


def get_latest_model_uri():
    # MLflow 3 stores models as Logged Models (models:/<model_id>),
    # not as a 'model' artifact inside the run.
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            order_by=[{"field_name": "creation_time", "ascending": False}],
            max_results=1,
            output_format="list",
        )
        if models:
            return f"models:/{models[0].model_id}"
    except Exception:
        pass
    return None


def evaluate():
    # Determine Model URI
    # Priority: Env Var > Latest Logged Model > Placeholder
    env_uri = os.getenv("MLFLOW_MODEL_URI_OVERRIDE")
    if env_uri:
        model_uri = env_uri
    else:
        latest_uri = get_latest_model_uri()
        if latest_uri:
            print(f"Auto-detected latest model: {latest_uri}")
            model_uri = latest_uri
        else:
            model_uri = "models:/<REPLACE_WITH_YOUR_MODEL_ID>"

    print(f"Loading test data from {DATA_PATH}...")
    _, X_test, _, y_test = get_train_test_split_data(DATA_PATH)

    # Combine for mlflow.evaluate
    eval_data = X_test.copy()
    eval_data["Churn"] = y_test

    print(f"Evaluating model: {model_uri}")

    # Insert your code here
    # Ensure we log to the same experiment as training
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Model_Evaluation"):
        # Use mlflow.models.evaluate
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets="Churn",
            model_type="classifier",
            evaluators=["default"],
        )

        print("\nEvaluation metrics logged to MLflow:")
        # Print a clean subset of metrics
        metrics_to_show = ["accuracy_score", "f1_score", "roc_auc"]
        clean_metrics = {
            k: v for k, v in result.metrics.items() if k in metrics_to_show
        }
        print(clean_metrics)


if __name__ == "__main__":
    evaluate()
