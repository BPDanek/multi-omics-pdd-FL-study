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

import json
from typing import Optional, Tuple

from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.sklearn.data_loader import load_data_for_range
import pandas as pd

def read_multi_modality_dataset(data_path, start: int=None, end: int=None):
    
    if start and end:
        data = pd.read_hdf(data_path, start=start, stop=end)
    else:
        data = pd.read_hdf(data_path) # validation set does not need to specify boundaries

    if ('ID' in data.columns):
        x = data.drop(columns=['ID', 'PHENO']).copy()
    else:
        x = data.drop(columns=['PHENO']).copy()

    y = pd.DataFrame(data['PHENO'].copy())    
    return x.to_numpy(), y.to_numpy().reshape((-1))

class SVMLearner(Learner):
    def __init__(
        self,
        data_split_filename,
        client_id,
        random_state: int = None,
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

        self.train_data = None
        self.valid_data = None
        self.n_samples = None
        self.svm = None
        self.kernel = None
        self.params = {}
    
    def load_data(self) -> dict:
        with open(self.data_split_filename) as file:
            data_split = json.load(file)

        data_path = data_split["data_path"]
        data_index = data_split["data_index"]

        # check if site_id and "valid" in the mapping dict
        if self.client_id not in data_index.keys():
            raise ValueError(
                f"Dict of data_index does not contain Client {self.client_id} split",
            )

        if "valid_path" not in data_split.keys():
            raise ValueError(
                "Data does not contain Validation split",
            )
        
        valid_path = data_split["valid_path"]

        site_index = data_index[self.client_id]

        # training
        X_train, y_train = read_multi_modality_dataset(
            data_path=data_path, start=site_index["start"], end=site_index["end"]
        )

        # validation
        X_valid, y_valid = read_multi_modality_dataset(
            data_path=valid_path
        )

        train_data = (X_train, y_train, len(y_train))
        valid_data = (X_valid, y_valid, len(y_valid))
        
        return {"train": train_data, "valid": valid_data}

    def initialize(self, fl_ctx: FLContext):
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]
        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential use in aggregation
        self.n_samples = data["train"][-1]
        # model will be created after receiving global parameter of kernel

    def train(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        if curr_round == 0:
            # only perform training on the first round
            (x_train, y_train, train_size) = self.train_data
            self.kernel = global_param["kernel"]
            self.svm = SVC(kernel=self.kernel)
            # train model
            self.svm.fit(x_train, y_train)
            # get support vectors
            index = self.svm.support_
            local_support_x = x_train[index]
            local_support_y = y_train[index]
            self.params = {"support_x": local_support_x, "support_y": local_support_y}
        elif curr_round > 1:
            self.system_panic("Federated SVM only performs training for one round, system exiting.", fl_ctx)
        return self.params, self.svm

    def validate(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        # local validation with global support vectors
        # fit a standalone SVM with the global support vectors
        svm_global = SVC(kernel=self.kernel)
        support_x = global_param["support_x"]
        support_y = global_param["support_y"]
        svm_global.fit(support_x, support_y)
        # validate global model
        (x_valid, y_valid, valid_size) = self.valid_data
        y_pred = svm_global.predict(x_valid)
        auc = roc_auc_score(y_valid, y_pred)
        self.log_info(fl_ctx, f"AUC {auc:.4f}")
        metrics = {"AUC": auc}
        return metrics, svm_global

    def finalize(self, fl_ctx: FLContext) -> None:
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")