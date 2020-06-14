import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from loan_model import pipeline
from config import config
from config import model_config
from pipeline import loan_pipe
from processing.data_management import load_dataset, dataset_location
# from loan_model import __version__ as _version

import logging
_logger = logging.getLogger(__name__)

import mlflow
# import mlflow.xgboost
import mlflow.sklearn

mlflow.set_tracking_uri(config.MLFLOW_REMOTE_SERVER_URI)

def run_training() -> None:
    """Train the model."""

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

    loan_pipe.fit(X_train[config.FEATURES], y_train)

    # Log dataset and parameters to mlflow
    dataset_full_path_name = dataset_location()
    mlflow.set_tag("dataset", dataset_full_path_name)

    mlflow.log_param("penalty", model_config.PENALTY)
    mlflow.log_param("dual", model_config.DUAL)
    mlflow.log_param("C", model_config.C)
    mlflow.log_param("fit_intercept", model_config.FIT_INTERCEPT)
    mlflow.log_param("random_state", model_config.RANDOM_STATE)
    mlflow.log_param("class_weight", model_config.CLASS_WEIGHT)
    mlflow.log_param("max_iter", model_config.MAX_ITER)
    mlflow.log_param("multi_class", model_config.MULTI_CLASS)

    # Log the metrics
    score = loan_pipe.score(X_train, y_train)
    mlflow.log_metric("score", score)

    # Save the sklearn pipeline as mlflow model
    conda_env = mlflow.sklearn.get_default_conda_env()
    mlflow.sklearn.log_model(loan_pipe, "sklearn_pipeline", conda_env=conda_env)

if __name__ == "__main__":
    run_training()
