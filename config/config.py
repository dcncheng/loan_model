import pathlib

# import loan_model

# PACKAGE_ROOT = pathlib.Path(loan_model.__file__).resolve().parent
TRAINED_MODEL_DIR = "trained_models"

# DATASET_DIR = PACKAGE_ROOT / "datasets"
DATASET_DIR = "s3://cs-mlflow-artifact-store/datasets/loan_model"
S3_MODEL_PATH = "s3://cs-mlflow-artifact-store/Mlflow/1/75487555f27441a7a991f28132797549/artifacts/sklearn_pipeline"
EXPERIMENT_NAME = "loan_model"

# MLFLOW
MLFLOW_REMOTE_SERVER_URI = "http://104.210.54.211:5000"

# data
TRAINING_DATA_FILE = "lending_club_loan_data.csv"
TESTING_DATA_FILE = "lending_club_loan_test.csv"

PRE_TARGET = "loan_status"
TARGET = "loan_repaid"

# variables
FEATURES = [
    'loan_amnt', 
    'term', 
    'int_rate', 
    'installment', 
    'grade', 
    'sub_grade',
    'emp_title', 
    'emp_length', 
    'home_ownership', 
    'annual_inc', 
    'verification_status', 
    'issue_d', 
    'purpose', 
    'title', 
    'dti', 
    'earliest_cr_line', 
    'open_acc', 
    'pub_rec', 
    'revol_bal', 
    'revol_util', 
    'total_acc', 
    'initial_list_status', 
    'application_type',
    'mort_acc', 
    'pub_rec_bankruptcies', 
    'address'
]

DROP_FEATURES_PREPROCESS = [
    "emp_title", 
    "emp_length", 
    "title",
    "grade",
    "issue_d"
]


DROP_FEATURES_POSTPROCESS = [
    "loan_status",
    "earliest_cr_line",
    "address"
]


DROP_FEATURES = [
    "emp_title", 
    "emp_length", 
    "title",
    "grade",
    "issue_d",
    "earliest_cr_line",
    "address"
]

# numerical variables for transforming mort_acc
NUMERICAL_VARS_FOR_MORT_ACC = [
    "mort_acc",
    "total_acc"
]

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = [
    "revol_util",
    "pub_rec_bankruptcies",
    "mort_acc"
]

# categorical variable - term
# convert from string into new numerical
CATEGORICAL_VARS_TERM = [
    "term"
]

# categorical variable - address
# convert the initial feature 'address' into new feature 'zip_code'
CATEGORICAL_VARS_ADDRESS = [
    "address",
    "zip_code"
]

# categorical variable - earliest_cr_line
# convert the initial feature 'earliest_cr_line' 
# into new numerical feature 'earliest_cr_year'
CATEGORICAL_VARS_EARLIEST_CR_LINE = [
    "earliest_cr_line",
    "earliest_cr_year"
]

# cagegorical variables required imputation.
CATEGORICAL_VARS_IMPUTED = [
]

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = [
    "emp_title",
    "emp_length",
    "title"
]

# variables to log transform
NUMERICAL_LOG_VARS = []

# categorical variables to encode
CATEGORICAL_VARS_DERIVED = [
    "zip_code"
]
CATEGORICAL_VARS_ENCODED = [
    "sub_grade",
    "verification_status",
    "application_type",
    "initial_list_status", 
    "purpose",
    "home_ownership"
] + CATEGORICAL_VARS_DERIVED


# initial categorical variables
CATEGORICAL_VARS_INITIAL = [
    'term', 
    'grade', 
    'sub_grade',
    'emp_title', 
    'emp_length', 
    'home_ownership', 
    'verification_status', 
    'issue_d', 
    'purpose', 
    'title', 
    'earliest_cr_line', 
    'initial_list_status', 
    'application_type',
    'address'
]


NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS_INITIAL + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS_INITIAL if feature not in CATEGORICAL_VARS_WITH_NA
]


PIPELINE_NAME = "loan_model"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05