import os
import sys
from pathlib import Path

REPO_PARENT = Path(__file__).parents[3]

sys.path.append(os.path.abspath("."))
sys.path.append(str(REPO_PARENT))

import xgboost as xgb
import numpy as np

from federated_learning_multi_modality_ancestry.multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration

def test_fed_rfxgb_experiment(current_experiment: GlobalExperimentsConfiguration, model_path: str, fold_idx: int, num_clients: int, split_method: str, num_rounds: int, stratified: bool, num_local_rounds: int, proximal_mu: float, client_lr: float):

    num_trees = 100 # hyper-param?
    param = {}
    param["objective"] = "binary:logistic"
    param["eta"] = 0.1
    param["max_depth"] = 8
    param["eval_metric"] = "auc"
    param["nthread"] = 16
    param["num_parallel_tree"] = num_trees


    for name, dataset in current_experiment.get_combined_test_dataset():
        
        # if name == 'external test' and fold_idx != 0:
        #     continue

        X, y = current_experiment.as_features_labels(dataset, current_experiment.LABEL_COL)

        dmat = xgb.DMatrix(X, label=y)

        # validate model performance
        bst = xgb.Booster(param, model_file=model_path)
        # y_pred = bst.predict(dmat)
        # y_pred = np.rint(y_pred)

        # print(y)
        # print(y_pred)
        y_pred_proba = bst.predict(dmat)
        print(y)
        print(y_pred_proba)

        current_experiment.log_raw_experiment_results(
            fold_idx=fold_idx,
            algorithm_name='FedAvg XGBRFClassifier', 
            num_clients=num_clients, 
            split_method=split_method,
            stratified=stratified,
            val_name=name,
            num_rounds=num_rounds,
            num_local_rounds=num_local_rounds,
            client_lr=client_lr,
            proximal_mu=proximal_mu,
            y_true=y, 
            y_pred=y_pred_proba
        )

        current_experiment.log_runtime(fold_idx, 'FedAvg XGBRFClassifier', 'timer', current_experiment.get_time())
