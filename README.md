# loan_model

# To run from git
mlflow run --experiment-name=<experiment name> <git path>
- In case of use remote tracking server, ensure you set an environment variable 
export MLFLOW_TRACKING_URI=<server URI>

export MLFLOW_TRACKING_URI=http://104.210.54.211:5000/
mlflow run --experiment-name=loan_model git@github.com:dcncheng/loan_model.git
