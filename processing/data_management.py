import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from loan_model.config import config
from loan_model import __version__ as _version

import mlflow
import mlflow.sklearn

import logging
_logger = logging.getLogger(__name__)

import boto3
client = boto3.client('s3')

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return _data

def dataset_location() -> str:
    file_name=config.TRAINING_DATA_FILE
    full_path = f"{config.DATASET_DIR}/{file_name}"
    return full_path

def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    # Prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline

def load_mlflow_model() -> Pipeline:
    file_path = config.S3_MODEL_PATH
    _logger.info(f"=== Load model pipeline from S3 bucket: {file_path}")
    saved_pipeline = mlflow.sklearn.load_model(file_path)
    return saved_pipeline

def load_current_pipeline() -> Pipeline:
    current_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    current_path = config.TRAINED_MODEL_DIR / current_file_name
    pipeline = joblib.load(filename=current_path)
    return pipeline


def remove_old_pipelines(*, files_to_keep) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()
