import os
import sys
from pathlib import Path

REPO_PARENT = Path(__file__).parents[3]

sys.path.append(os.path.abspath("."))
sys.path.append(str(REPO_PARENT))

from typing import Any, Callable
import pandas as pd
import shutil
import logging
# from nvflare.apis.fl_constant import JobConstants 
# from sklearn.metrics import roc_auc_score
# from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner
from joblib import load
from sklearn.linear_model import SGDClassifier
import numpy as np

from federated_learning_multi_modality_ancestry.multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration

logging.basicConfig(level=logging.ERROR)

# test simulation
def test_fed_linear_experiment(current_experiment: GlobalExperimentsConfiguration, model_path: str, fold_idx: int, num_clients: int, split_method: str, num_rounds: int, stratified: bool, num_local_rounds: int, proximal_mu: float, client_lr: float):
    model_params = load(model_path)
    clf = SGDClassifier(random_state=current_experiment.RANDOM_SEED, loss="log_loss")
    clf.coef_ = model_params['coef']
    clf.intercept_ = model_params['intercept']
    clf.classes_ = model_params['n_classes']

    for name, dataset in current_experiment.get_combined_test_dataset():
        
        # if name == 'external test' and fold_idx != 0:
        #     continue


        X, y = current_experiment.as_features_labels(dataset, current_experiment.LABEL_COL)

        # validate model performance
        # y_pred = clf.predict(X)
        y_pred_proba = clf.predict_proba(X)
        
        print(y)
        print(y_pred_proba)
        # print(y_pred)
        # y_pred = np.rint(y_pred)
        # print(sum(y_pred) == len(y_pred))
        # print(sum(y) == len(y))

        current_experiment.log_raw_experiment_results(
            fold_idx=fold_idx,
            algorithm_name='FedAvg SGDClassifier', 
            num_clients=num_clients, 
            split_method=split_method,
            stratified=stratified,
            val_name=name,
            num_rounds=num_rounds,
            num_local_rounds=num_local_rounds,
            client_lr=client_lr,
            proximal_mu=proximal_mu,
            y_true=y,
            y_pred=y_pred_proba[:, 1]
        )

        current_experiment.log_runtime(fold_idx, 'FedAvg SGDClassifier', 'timer', current_experiment.get_time())
