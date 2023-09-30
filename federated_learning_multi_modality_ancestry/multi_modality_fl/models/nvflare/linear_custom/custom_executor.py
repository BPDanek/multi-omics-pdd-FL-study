# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import logging
from typing import Optional, Tuple

import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier

import pandas as pd 

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner

def read_multi_modality_dataset(data_path):
    
    data = pd.read_hdf(data_path) # validation set does not need to specify boundaries

    if ('ID' in data.columns):
        x = data.drop(columns=['ID', 'PHENO']).copy().to_numpy()
    else:
        x = data.drop(columns=['PHENO']).copy().to_numpy()

    y = pd.DataFrame(data['PHENO'].copy()).to_numpy().reshape((-1, ))

    return x, y

def computeAUCPR(y_true, y_pred):
    # print("yp", y_pred.shape)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred[:, 1])
    return metrics.auc(recall, precision)

class LinearLearner(Learner):
    def __init__(
        self,
        data_split_filename,
        client_id,
        random_state: int,
        learning_rate: float,
    ):
        super().__init__()
        self.data_split_filename = data_split_filename
        self.client_id = client_id

        self.random_state = random_state
        self.train_data = None
        self.valid_data = None
        self.n_samples = None
        self.local_model = None
        self.n_features = None
        self.learning_rate = learning_rate
    
    def load_data(self) -> dict:
        with open(self.data_split_filename) as file:
            data_split = json.load(file)

        data_path = data_split["data_path"]        
        valid_path = data_split["valid_path"]

        # training
        X_train, y_train = read_multi_modality_dataset(data_path=data_path)

        # validation
        X_valid, y_valid = read_multi_modality_dataset(data_path=valid_path)

        train_data = (X_train, y_train, len(y_train))
        valid_data = (X_valid, y_valid, len(y_valid))

        print(f"INFO: Loaded data from {data_path} and {valid_path}")
        
        return {"train": train_data, "valid": valid_data}

    def initialize(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Loading data from {self.data_split_filename}")
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]

        assert set(pd.DataFrame(self.train_data[0]).itertuples(index=False, name=None)) & set(pd.DataFrame(self.valid_data[0]).itertuples(index=False, name=None)) == set()

        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential aggregation
        self.n_samples = data["train"][-1]
        self.n_features = data["train"][0].shape[1]
        # model will be created after receiving global parameters

    def set_parameters(self, params):
        self.local_model.coef_ = params["coef"]
        if self.local_model.fit_intercept:
            self.local_model.intercept_ = params["intercept"]

    def train(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        (x_train, y_train, train_size) = self.train_data
        if curr_round == 0:
            # initialize model with global_param
            # and set to all zero
            fit_intercept = bool(global_param["fit_intercept"])
            self.local_model = SGDClassifier(
                # loss=global_param["loss"],
                loss="log_loss",
                penalty=global_param["penalty"],
                fit_intercept=fit_intercept,
                learning_rate=global_param["learning_rate"],
                eta0=global_param["eta0"],
                early_stopping=True,
                max_iter=global_param["max_iter"], 
                warm_start=True,
                random_state=self.random_state,
            )
            n_classes = global_param["n_classes"]
            self.local_model.classes_ = np.array(list(range(n_classes)))
            self.local_model.coef_ = np.zeros((1, self.n_features))
            if fit_intercept:
                self.local_model.intercept_ = np.zeros((1, ))
        
        # Training starting from global model
        # Note that the parameter update using global model has been performed
        # during global model evaluation

        y_pred = self.local_model.predict_proba(x_train)
        aucpr = computeAUCPR(y_train, y_pred)
        self.log_info(fl_ctx, f"Untrained training aucpr {aucpr:.4f}")

        self.local_model.fit(x_train, y_train)

        y_pred = self.local_model.predict_proba(x_train)
        print("linear y_pred", y_pred.shape)
        aucpr = computeAUCPR(y_train, y_pred)

        self.log_info(fl_ctx, f"Trained training aucpr {aucpr:.4f}")
        
        assert set(pd.DataFrame(self.train_data[0]).itertuples(index=False, name=None)) & set(pd.DataFrame(self.valid_data[0]).itertuples(index=False, name=None)) == set()
        
        if self.local_model.fit_intercept:
            params = {
                "coef": self.local_model.coef_,
                "intercept": self.local_model.intercept_,
                "n_classes": self.local_model.classes_,
            }
        else:
            params = {"coef": self.local_model.coef_}
        return copy.deepcopy(params), self.local_model

    def validate(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        # set local model with global parameters
        self.set_parameters(global_param)
        # perform validation
        (x_valid, y_valid, valid_size) = self.valid_data
        y_pred = self.local_model.predict_proba(x_valid)
        
        
        aucpr = computeAUCPR(y_valid, y_pred)
        self.log_info(fl_ctx, f"Validation aucpr {aucpr:.4f}")
        metrics = {"aucpr": aucpr}

        assert set(pd.DataFrame(self.train_data[0]).itertuples(index=False, name=None)) & set(pd.DataFrame(self.valid_data[0]).itertuples(index=False, name=None)) == set()

        return metrics, self.local_model

    def finalize(self, fl_ctx: FLContext):
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")