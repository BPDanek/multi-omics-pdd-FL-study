import json

import pandas as pd
import xgboost as xgb

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.xgboost.tree_based.executor import FedXGBTreeExecutor

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

class FedXGBTreeExecutor_multi_modality(FedXGBTreeExecutor):
    def __init__(
        self,
        data_split_filename,
        training_mode,
        lr_scale,
        num_client_bagging: int = 1,
        lr_mode: str = "uniform",
        local_model_path: str = "model.json",
        global_model_path: str = "model_global.json",
        learning_rate: float = 0.1,
        objective: str = "binary:logistic",
        num_local_parallel_tree: int = 1,
        local_subsample: float = 1,
        max_depth: int = 8,
        eval_metric: str = "auc",
        nthread: int = 16,
        tree_method: str = "hist",
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__(
            training_mode=training_mode,
            num_client_bagging=num_client_bagging,
            lr_scale=lr_scale,
            lr_mode=lr_mode,
            local_model_path=local_model_path,
            global_model_path=global_model_path,
            learning_rate=learning_rate,
            objective=objective,
            num_local_parallel_tree=num_local_parallel_tree,
            local_subsample=local_subsample,
            max_depth=max_depth,
            eval_metric=eval_metric,
            nthread=nthread,
            tree_method=tree_method,
            train_task_name=train_task_name,
        )
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