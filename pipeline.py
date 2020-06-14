import pandas as pd
import numpy as mp
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler 

from processing import preprocessors as pp
from config import config
from config import model_config

import logging
_logger = logging.getLogger(__name__)

loan_pipe = Pipeline(
    [        
        (
            "numerical_mort_acc_transform",
            pp.NumericalMortAccImputer(variables=config.NUMERICAL_VARS_FOR_MORT_ACC),
        ),
        (
            "numerical_inputer",
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA),
        ),
        (   "categorical_term_transformer",
            pp.CategoricalTermTransformer(variables=config.CATEGORICAL_VARS_TERM),
        ),
        (
            "categorical_earliest_cr_line_transformer",
            pp.CategoricalEarliestCrLineTransformer(variables=config.CATEGORICAL_VARS_EARLIEST_CR_LINE),
        ),
        (
            "categorical_address_transformer",
            pp.CategoricalAddressTransformer(variables=config.CATEGORICAL_VARS_ADDRESS),
        ),
        (
            "categorical_imputer",
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_IMPUTED),
        ),
        (
            "categorical_rare_label_encoder",
            pp.RareLabelCategoricalEncoder(tol=0.01, variables=config.CATEGORICAL_VARS_ENCODED),
        ),
        (
            "categorical_encoder",
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS_ENCODED),
        ),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES),
        ),
        ("scaler", MinMaxScaler()),
        ("logistic_model", LogisticRegression(penalty=model_config.PENALTY,
                                            dual=model_config.DUAL,
                                            C=model_config.C,
                                            fit_intercept=model_config.FIT_INTERCEPT,
                                            random_state=model_config.RANDOM_STATE,
                                            class_weight=model_config.CLASS_WEIGHT,
                                            max_iter=model_config.MAX_ITER,
                                            multi_class=model_config.MULTI_CLASS)),
    ]
)
