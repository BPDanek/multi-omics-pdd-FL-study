import itertools
import json
import os
import shutil
import sys
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.ERROR)

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./federated_learning_multi_modality_ancestry'))

from multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration, write_json
from multi_modality_fl.experiments_code.baseline import run_baseline_experiments

from multi_modality_fl.experiments_code.test_nvflare_fed_linear import test_fed_linear_experiment
from multi_modality_fl.experiments_code.nvflare_fed_linear import run_fed_linear_experiments

from multi_modality_fl.experiments_code.nvflare_fed_rfxgb import run_fed_rfxgb_experiment
from multi_modality_fl.experiments_code.test_nvflare_fed_rfxgb import test_fed_rfxgb_experiment

from multi_modality_fl.experiments_code.flwr_fed_logreg import run_fed_logreg_experiment

from multi_modality_fl.experiments_code.flwr_fed_mlp import run_fed_mlp_experiment

import errno
import os
import signal
import functools


REPO_PARENT = '/Users/benjamindanek/Downloads/cell_patterns_submission_materials'
DATASET_PATH = '/Users/benjamindanek/Downloads/cell_patterns_submission_materials/not_for_submission/data'


logging.basicConfig(filename='my_log_file.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeoutError(Exception):
    pass

def timeout(seconds=3, error_message=os.strerror(errno.ETIME)):
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

# each of 6 client should run for 2 minutes max, so if we deviate from that by 3 minutes we retry
# @timeout(2*6*60 + 3*60)
def run_exp(write_results_path, repo_parent, fold_idx, num_rounds, num_local_rounds, client_lr, site_config, split_method, stratified):
    
    # experiments are deterministic
    current_experiment = GlobalExperimentsConfiguration(
        base_path=os.path.join(os.getcwd(), os.path.join(
            'multi_modality_fl', 'experiments')),
        experiment_name='global_experiment_runner',
        random_seed=0
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

    run_fed_logreg_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)
    run_fed_mlp_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)

    proximal_mu = 0.5
    run_fed_logreg_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
    run_fed_mlp_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)

    proximal_mu = 2.0
    run_fed_logreg_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
    run_fed_mlp_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)

    proximal_mu = 5.0
    run_fed_logreg_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
    run_fed_mlp_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)

    proximal_mu = 8.0
    run_fed_logreg_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
    run_fed_mlp_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)

    write_paths = current_experiment.write_raw_experiment_results_as_df(fold_idx, write_results_path)
    print(write_paths)

def run_baseline_exp(write_results_path, fold_idx):
    # experiments are deterministic
    current_experiment = GlobalExperimentsConfiguration(
        base_path=os.path.join(os.getcwd(), os.path.join(
            'multi_modality_fl', 'experiments')),
        experiment_name='global_experiment_runner',
        random_seed=0
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


# delete directories which persist data
def delete_dir():
    dir =  os.path.join(REPO_PARENT, 'federated_learning_multi_modality_ancestry/multi_modality_fl/experiments')
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
    write_results_path=os.path.join(REPO_PARENT, 'federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_uniform_strat/')
    delete_dir()
    run_baseline_exp(write_results_path, fold_idx) # next
    delete_dir()
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 2, split_method, stratified)
    delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 4, split_method, stratified)
    # delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 6, split_method, stratified)
    # delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 8, split_method, stratified)
    # delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 10, split_method, stratified)
    # delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 12, split_method, stratified)
    # delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 14, split_method, stratified)
    # delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 16, split_method, stratified)
    # delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 18, split_method, stratified)
    # delete_dir()


    split_method = 'uniform'
    stratified = False
    # records experiment results
    write_results_path=os.path.join(REPO_PARENT, 'federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_uniform_non_strat/')
    delete_dir()
    run_baseline_exp(write_results_path, fold_idx)
    delete_dir()
    run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 2, split_method, stratified)
    delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 4, split_method, stratified)
    # delete_dir()
    
    # # the below fails sometimes because there are not enough samples to run the local learners. Not included in paper
    # # run_exp(write_results_path, repo_parent, fold_idx, num_round, num_local_rounds, client_lr, 6, split_method, stratified)
    # # delete_dir()


    split_method = 'linear'
    stratified = False
    # records experiment results
    write_results_path=os.path.join(REPO_PARENT, 'federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_linear_non_strat/')
    delete_dir()
    run_baseline_exp(write_results_path, fold_idx) # next
    delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 2, split_method, stratified)
    # delete_dir()
    # run_exp(write_results_path, REPO_PARENT, fold_idx, num_round, num_local_rounds, client_lr, 4, split_method, stratified)
    # delete_dir()

if __name__ == "__main__":

    run_series(0)
    # run_series(1)
    # run_series(2)
    # run_series(3)
    # run_series(4)
    # run_series(5)
