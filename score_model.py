import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from loan_model.processing.data_management import load_mlflow_model
from loan_model.config import config
from loan_model.processing.data_management import load_dataset, load_current_pipeline, dataset_location

import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri(config.MLFLOW_REMOTE_SERVER_URI)

def run_scoring() -> None:
    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # transform the target
    data[config.TARGET] = data[config.PRE_TARGET].map({'Fully Paid':1, 'Charged Off':0})
    data = data.drop(config.PRE_TARGET, axis=1)

    # divide train and test, set random_state
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )  

    # Set MLFlow experiment
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # Load the model from MLflow artifact store
    mlflow_pipeline = load_mlflow_model()
    score = mlflow_pipeline.score(X_train, y_train)

    mlflow.log_metric("score", score)


if __name__ == "__main__":
    run_scoring()
