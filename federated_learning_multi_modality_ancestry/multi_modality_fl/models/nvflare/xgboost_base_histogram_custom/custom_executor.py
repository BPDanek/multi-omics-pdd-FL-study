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

import pandas as pd
import xgboost as xgb

from nvflare.app_opt.xgboost.histogram_based.executor import FedXGBHistogramExecutor


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
    return x.to_numpy(), y.to_numpy().reshape((-1, ))


class FedXGBHistogramExecutor_multi_modality(FedXGBHistogramExecutor):
    def __init__(self, data_split_filename, num_rounds, early_stopping_round, xgboost_params, verbose_eval=False):
        """Federated XGBoost Executor for histogram-base collaboration.

        Args:
            data_split_filename: file name to data splits
            num_rounds: number of boosting rounds
            early_stopping_round: early stopping round
            xgboost_params: parameters to passed in xgb
            verbose_eval: verbose_eval in xgb
        """
        super().__init__(num_rounds, early_stopping_round, xgboost_params, verbose_eval)
        self.data_split_filename = data_split_filename

    def load_data(self):
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

        dmat_train = xgb.DMatrix(X_train, label=y_train)
        print("dmt", (dmat_train))
        # validation
        X_valid, y_valid = read_multi_modality_dataset(
            data_path=valid_path
        )
        dmat_valid = xgb.DMatrix(X_valid, label=y_valid)
        print("dmt val", (dmat_valid))

        return dmat_train, dmat_valid
