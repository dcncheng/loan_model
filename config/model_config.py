
# Deployment
DEPLOYMENT_NAME = "loan_model"

#
# Azure deployment
#
WORKSPACE_NAME = "AIML-Workspace"
SUBSCRIPTION_ID = "ad8894e8-bf86-449c-9165-3c3c572dccc6"
RESOURCE_GROUP = "AIML-Research"
LOCATION = "Central US"

# 
# Sklearn parameters
# 
PENALTY = 'l2'
DUAL = False
C = 1.0
FIT_INTERCEPT = True
RANDOM_STATE = 11
CLASS_WEIGHT = 'balanced'
MAX_ITER = 120
MULTI_CLASS = 'auto'


#
# XgBoost parameters
#
def get_xgb_params():
    params = {
        'objective': 'multi:softprob',
        'num_class': 2,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 1.0,
        'subsample': 1.0,
        'seed': 101,
    }
    return params

OBJECTIVE = 'binary:logistic'
MAX_DEPTH = 10
LEARNING_RATE = 0.3

# subsample: [0, 1]
SUBSAMPLE = 0.8
REG_ALPHA = 0