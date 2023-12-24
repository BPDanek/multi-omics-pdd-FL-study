import os
import sys
from pathlib import Path
import time

import numpy as np

REPO_PARENT = Path(__file__).parents[3]

sys.path.append(os.path.abspath("."))
sys.path.append(str(REPO_PARENT))

import shutil
import errno
import signal
import functools
import logging
import argparse

from federated_learning_multi_modality_ancestry.multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration
from federated_learning_multi_modality_ancestry.multi_modality_fl.experiments_code.baseline import run_baseline_experiments
from federated_learning_multi_modality_ancestry.multi_modality_fl.experiments_code.test_nvflare_fed_linear import test_fed_linear_experiment
from federated_learning_multi_modality_ancestry.multi_modality_fl.experiments_code.nvflare_fed_linear import run_fed_linear_experiments
from federated_learning_multi_modality_ancestry.multi_modality_fl.experiments_code.nvflare_fed_rfxgb import run_fed_rfxgb_experiment
from federated_learning_multi_modality_ancestry.multi_modality_fl.experiments_code.test_nvflare_fed_rfxgb import test_fed_rfxgb_experiment
from federated_learning_multi_modality_ancestry.multi_modality_fl.experiments_code.flwr_fed_logreg import run_fed_logreg_experiment
from federated_learning_multi_modality_ancestry.multi_modality_fl.experiments_code.flwr_fed_mlp import run_fed_mlp_experiment

logging.basicConfig(level=logging.ERROR)
logging.basicConfig(filename='my_log_file.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset path")

args = parser.parse_args()

DATASET_PATH = args.dataset

class TimeoutError(Exception):
    pass

def timeout(seconds=3, error_message=os.strerror(errno.ETIME)):
    """
    Decorator to timeout a function if it runs for too long.
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

# The timeout utility can be used to cap the runtime of experiments
# each of 6 client should run for 2 minutes max, so if we deviate from that by 3 minutes we retry
# @timeout(2*6*60 + 3*60)
def run_exp(write_results_path, REPO_PARENT, fold_idx, num_rounds, num_local_rounds, client_lr, site_config, split_method, stratified):
    """
    Run the array of FL experiments
    """

    # define experiments configurations so they are deterministic
    current_experiment = GlobalExperimentsConfiguration(
        base_path = str(REPO_PARENT / "federated_learning_multi_modality_ancestry" / "multi_modality_fl" / "experiments"),
        experiment_name = 'global_experiment_runner',
        random_seed = 0
    )

    current_experiment.initialize_data_splits(
        dataset_folder=DATASET_PATH,
        dataset=GlobalExperimentsConfiguration.MULTIMODALITY,
        split_method=GlobalExperimentsConfiguration.SKLEARN
    )

    current_experiment.set_fold(fold_idx=fold_idx)

    print("running:", num_rounds, num_local_rounds, client_lr, site_config, split_method, stratified)
    model_path = run_fed_linear_experiments(current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)            
    test_fed_linear_experiment(current_experiment, model_path, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)

    model_path = run_fed_rfxgb_experiment(current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)
    test_fed_rfxgb_experiment(current_experiment, model_path, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)

    run_fed_logreg_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)
    run_fed_mlp_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)

    proximal_mu = 0.5
    run_fed_logreg_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
    run_fed_mlp_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)

    proximal_mu = 2.0
    run_fed_logreg_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
    run_fed_mlp_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)

    proximal_mu = 5.0
    run_fed_logreg_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
    run_fed_mlp_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)

    proximal_mu = 8.0
    run_fed_logreg_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
    run_fed_mlp_experiment(REPO_PARENT, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)

    write_paths = current_experiment.write_raw_experiment_results_as_df(fold_idx, write_results_path)
    print(write_paths)

    current_experiment.write_runtime_logs(write_results_path)
    

def run_baseline_exp(write_results_path, fold_idx):
    """
    Run the array of central experiments
    """
    # define experiments configuration so they are deterministic
    current_experiment = GlobalExperimentsConfiguration(
        base_path = str(REPO_PARENT / "federated_learning_multi_modality_ancestry" / "multi_modality_fl" / "experiments"),
        experiment_name = 'global_experiment_runner',
        random_seed = 0
    )

    current_experiment.initialize_data_splits(
        dataset_folder=DATASET_PATH,
        dataset=GlobalExperimentsConfiguration.MULTIMODALITY,
        split_method=GlobalExperimentsConfiguration.SKLEARN
    )

    current_experiment.set_fold(fold_idx=fold_idx)

    # performs logging
    run_baseline_experiments(current_experiment, fold_idx)

    write_paths = current_experiment.write_raw_experiment_results_as_df(fold_idx, write_results_path)
    print(write_paths)
    current_experiment.write_runtime_logs(write_results_path)


def delete_dir(REPO_PARENT):
    """
    delete directories which persist data
    """
    dir =  str(REPO_PARENT / "federated_learning_multi_modality_ancestry" / "multi_modality_fl" / "experiments")
    if os.path.exists(dir):
        shutil.rmtree(dir)
    
    # delete directory
    dir = '/tmp/nvflare'
    if os.path.exists(dir):
        shutil.rmtree(dir)


def run_series(fold_idx: int):
    
    num_round = 5
    num_local_rounds = 300
    client_lr = 1e-3

    split_method = 'uniform'
    stratified = True
    # records experiment results
    write_results_path = str(REPO_PARENT / "federated_learning_multi_modality_ancestry" / "multi_modality_fl" / "results" / "experiment_logs" / "manual_experiments_uniform_strat")
    delete_dir(REPO_PARENT)
    run_baseline_exp(write_results_path, fold_idx) # next
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 2, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 4, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 6, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 8, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 10, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 12, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 14, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 16, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 18, split_method, stratified)
    delete_dir(REPO_PARENT)


    split_method = 'uniform'
    stratified = False
    # records experiment results
    write_results_path = str(REPO_PARENT / "federated_learning_multi_modality_ancestry" / "multi_modality_fl" / "results" / "experiment_logs" / "manual_experiments_uniform_non_strat")
    delete_dir(REPO_PARENT)
    run_baseline_exp(write_results_path, fold_idx)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 2, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 4, split_method, stratified)
    delete_dir(REPO_PARENT)
    
    # the below fails sometimes because there are not enough samples to run the local learners. Not included in paper
    # run_exp(write_results_path, repo_parent, fold_idx, num_round, num_local_rounds, client_lr, 6, split_method, stratified)
    # delete_dir(REPO_PARENT)


    split_method = 'linear'
    stratified = False
    # records experiment results
    write_results_path = str(REPO_PARENT / "federated_learning_multi_modality_ancestry" / "multi_modality_fl" / "results" / "experiment_logs" / "manual_experiments_linear_non_strat")
    delete_dir(REPO_PARENT)
    run_baseline_exp(write_results_path, fold_idx) # next
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 2, split_method, stratified)
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 4, split_method, stratified)
    delete_dir(REPO_PARENT)


def run_timer_series(fold_idx: int):
    
    num_round = 5
    num_local_rounds = 300
    client_lr = 1e-3

    split_method = 'uniform'
    stratified = True
    # records experiment results
    write_results_path = str(REPO_PARENT / "federated_learning_multi_modality_ancestry" / "multi_modality_fl" / "results" / "experiment_logs" / "timed_experiments_uniform_strat")
    delete_dir(REPO_PARENT)
    run_baseline_exp(write_results_path, fold_idx) # next
    delete_dir(REPO_PARENT)
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 2, split_method, stratified)
    delete_dir(REPO_PARENT)


def run_runtime_logger():
    """
    runtime logger to capture the runtime of different FL methods.
    """

    # if any of the K folds fails, then you can rerun the experiment for that fold only since folds are deterministic
    run_timer_series(0)
    run_timer_series(1)
    run_timer_series(2)
    run_timer_series(3)
    run_timer_series(4)
    run_timer_series(5)

def run_main_experiments():
    """
    Run the experiments for the main paper
    """
    # if any of the K folds fails, then you can rerun the experiment for that fold only since folds are deterministic
    run_series(0)
    run_series(1)
    run_series(2)
    run_series(3)
    run_series(4)
    run_series(5)

if __name__ == "__main__":

    run_main_experiments()
    run_runtime_logger() # todo: final run for K=5 folds