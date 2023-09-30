import itertools
import json
import os
import sys
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.ERROR)

sys.path.append(os.path.abspath('.'))

from multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration, write_json
from multi_modality_fl.experiments_code.baseline import run_baseline_experiments

from multi_modality_fl.experiments_code.test_nvflare_fed_linear import test_fed_linear_experiment
from multi_modality_fl.experiments_code.nvflare_fed_linear import run_fed_linear_experiments

from multi_modality_fl.experiments_code.nvflare_fed_rfxgb import run_fed_rfxgb_experiment
from multi_modality_fl.experiments_code.test_nvflare_fed_rfxgb import test_fed_rfxgb_experiment

from multi_modality_fl.experiments_code.flwr_fed_logreg import run_fed_logreg_experiment

from multi_modality_fl.experiments_code.flwr_fed_mlp import run_fed_mlp_experiment

repo_parent = '/Users/benjamindanek/Code/'

if __name__ == '__main__':
    current_experiment = GlobalExperimentsConfiguration(
        base_path=os.path.join(os.getcwd(), os.path.join(
            'multi_modality_fl', 'experiments')),
        experiment_name='global_experiment_runner',
        random_seed=0
    )

    current_experiment.initialize_data_splits(
        dataset_folder=f'{repo_parent}/federated_learning_multi_modality_ancestry/data',
        dataset=GlobalExperimentsConfiguration.MULTIMODALITY,
        split_method=GlobalExperimentsConfiguration.SKLEARN
    )

    hyper_params = {
        "num_rounds": [5],
        "num_local_rounds": [300],# [1, 300, 500],#, 10],
        "client_lr": [1e-3],  #[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "site_configurations": [2, 4, 6, 8, 10, 12],
        "split_methods": ["uniform"],#, "polynomial"],
        "stratified": [True],
    }
 
    experiments_to_run = list(itertools.product(hyper_params['num_rounds'], hyper_params['num_local_rounds'], hyper_params['client_lr'], hyper_params['site_configurations'], hyper_params['split_methods'], hyper_params['stratified']))

    write_json(experiments_to_run, f'{repo_parent}/federated_learning_multi_modality_ancestry/multi_modality_fl/results/experiments_to_run.json')

    print("\nExperiments to run: \n", "\n".join([str(x) for x in experiments_to_run]), "\n")
    try:
        for fold_idx in range(current_experiment.K):
            current_experiment.set_fold(fold_idx=fold_idx)

            run_baseline_experiments(current_experiment, fold_idx) # learning rate for sgd?
             
            for num_rounds, num_local_rounds, client_lr, site_config, split_method, stratified in tqdm(list(experiments_to_run), f"Training loop; k = {fold_idx}"):

                print("running:", num_rounds, num_local_rounds, client_lr, site_config, split_method, stratified)
                model_path = run_fed_linear_experiments(current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)            
                test_fed_linear_experiment(current_experiment, model_path, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)
                
                model_path = run_fed_rfxgb_experiment(current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)
                test_fed_rfxgb_experiment(current_experiment, model_path, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)

                run_fed_logreg_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)
                run_fed_mlp_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=None, client_lr=client_lr)
                
                for proximal_mu in [0.8]:
                    run_fed_logreg_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
                    run_fed_mlp_experiment(repo_parent, current_experiment, fold_idx, site_config, split_method, num_rounds=num_rounds, stratified=stratified, num_local_rounds=num_local_rounds, proximal_mu=proximal_mu, client_lr=client_lr)
                    # break
                # break
            # write raw experiment results 
        write_paths = current_experiment.write_raw_experiment_results_as_df(f'{repo_parent}/federated_learning_multi_modality_ancestry/multi_modality_fl/results/global_experiment_results')
        print(write_paths)
    except Exception as e:
        print(e)
        try: 
            current_experiment.write_raw_experiment_results_as_df(f'{repo_parent}/federated_learning_multi_modality_ancestry/multi_modality_fl/results/global_experiment_results_abort')
        except:
            with open(f'{repo_parent}/federated_learning_multi_modality_ancestry/multi_modality_fl/results/experiment_logs_abort', "w") as f:
                json.dump(current_experiment.raw_experiment_logs.values(), f, default=str)