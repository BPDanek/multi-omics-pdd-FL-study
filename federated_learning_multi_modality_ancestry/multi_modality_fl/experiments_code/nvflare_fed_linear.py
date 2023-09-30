
import os
from typing import Any, Callable
import pandas as pd
import shutil
import logging
from nvflare.apis.fl_constant import JobConstants 
from sklearn.metrics import roc_auc_score
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner
from joblib import load
from sklearn.linear_model import SGDClassifier
import sys
logging.basicConfig(level=logging.ERROR)

sys.path.append(os.path.abspath('.'))
from multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration, write_json, read_json

def run_fed_linear_experiments(current_experiment: GlobalExperimentsConfiguration, fold_idx: int, num_clients: int, split_method: str, num_rounds: int, stratified: bool, num_local_rounds: int, proximal_mu: float, client_lr: float):

    SERVER_KEY = "server"
    """server FL site"""
    SERVER_VAL = f"app_{SERVER_KEY}"
    """server FL app name"""

    def CLIENT_KEY(site_name_prefix, i):
        """client FL site"""
        return f"{site_name_prefix}{i}"

    def CLIENT_VAL(site_name_prefix, i): 
        """client FL app name"""
        return f"app_{site_name_prefix}{i}"

    def get_deploy_map(site_name_prefix: str, n_sites: int):
        """
        Generate a map of which apps in the job being uploaded will be deployed to which FL client sites.
        
        https://nvflare.readthedocs.io/en/main/real_world_fl/job.html#deploy-map
        """
        deploy_map = {SERVER_VAL: [SERVER_KEY]}
        for i in range(1, n_sites + 1):
            deploy_map[CLIENT_VAL(site_name_prefix, i)] = [CLIENT_KEY(site_name_prefix, i)]

        return deploy_map
    
    # define nvflare experiments as jobs
    ALL_JOBS_PATH = os.path.join(current_experiment.experiment_path, 'jobs')
    """The portion of the experiment data path which is reserved for nvflare job definitions"""

    # root for this series of jobs
    # It is convenient to conduct several experiments at a time, so this interface was developed. 
    JOB_BASE_FOLDER = 'linear_base'
    """The root of all jobs for the current experiment. (ie `random_forest_base`)"""

    # Base folder for jobs
    base_path = os.path.join(ALL_JOBS_PATH, JOB_BASE_FOLDER)
    print("using base path", base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    # create base job
    # we copy the base job when generating new jobs & change a few aspects in the design of experiemnts

    # 1. define meta
    base_job_meta_path = os.path.join(base_path, JobConstants.META_FILE)
    base_job_meta = {
    "name": "sklearn_linear",
    "resource_spec": {},
    "deploy_map": {
        "app": [
        "@ALL"
        ]
    }
    }
    # src from: https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/random_forest/jobs/random_forest_base/meta.json
    write_json(base_job_meta, base_job_meta_path)

    # 2. define server & client configs
    base_job_root = os.path.join(base_path, "app")
    base_job_configs = os.path.join(base_job_root, "config")
    if not os.path.exists(base_job_configs):
        os.makedirs(base_job_configs, exist_ok=True)

    # 2.1. define base job config for server
    BEST_MODEL_SAVE = "model_param.joblib"
    base_job_server_config_path = os.path.join(base_job_configs, JobConstants.SERVER_JOB_CONFIG)
    base_job_server_config = {
        "format_version": 2,
        "min_clients": num_clients,
        "num_rounds": num_rounds,
        "server": {
            "heart_beat_timeout": 600,
            "task_request_interval": 0.05
        },
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
            {
            "id": "persistor",
            "path": "nvflare.app_opt.sklearn.joblib_model_param_persistor.JoblibModelParamPersistor",
            "args": {
                "initial_params": {
                    "n_classes": 2,
                    "learning_rate": "adaptive",
                    "eta0": client_lr,
                    "max_iter": num_local_rounds,
                    "loss": "hinge",
                    "penalty": "l2",
                    "fit_intercept": 1,
                    "save_name": BEST_MODEL_SAVE
                    }
                },
            },
            {
            "id": "shareable_generator",
            "name": "FullModelShareableGenerator",
            "args": {}
            },
            {
            "id": "aggregator",
            "name": "InTimeAccumulateWeightedAggregator",
            "args": {
                "expected_data_kind": "WEIGHTS"
            }
            }
        ],
        "workflows": [
            {
            "id": "scatter_and_gather",
            "name": "ScatterAndGather",
            "args": {
                "min_clients" : -1,
                "num_rounds" : num_rounds,
                "start_round": 0,
                "wait_time_after_min_received": 0,
                "aggregator_id": "aggregator",
                "persistor_id": "persistor",
                "shareable_generator_id": "shareable_generator",
                "train_task_name": "train",
                "train_timeout": 0
            }
            }
        ]
        }
    # src: https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/random_forest/jobs/random_forest_base/app/config/config_fed_server.json
    write_json(base_job_server_config, base_job_server_config_path)

    # 2.2. define the base job config for a client
    base_job_client_config_path = os.path.join(base_job_configs, JobConstants.CLIENT_JOB_CONFIG)
    base_job_client_config = {
    "format_version": 2,

    "executors": [
        {
        "tasks": ["train"],
        "executor": {
            "id": "Executor",
            "path": "nvflare.app_opt.sklearn.sklearn_executor.SKLearnExecutor",
            "args": {
            "learner_id": "linear_learner"
            }
        }
        }
    ],
    "task_result_filters": [],
    "task_data_filters": [],
    "components": [
        {
        "id": "linear_learner",
        "path": "custom_executor.LinearLearner",
        "args": {
            "data_split_filename": "/tmp/",
            "client_id": "none-provided",
            "random_state": current_experiment.RANDOM_SEED,
            "learning_rate": client_lr
        }
        }
    ]
    }


    # src: https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/xgboost/histogram-based/jobs/base/app/config/config_fed_client.json
    write_json(base_job_client_config, base_job_client_config_path)

    # 3. copy over custom contents for this experiment
    # source
    custom_data = os.path.join(os.getcwd(), "multi_modality_fl", "models", "nvflare", "linear_custom")

    # destination
    base_job_custom = os.path.join(base_job_root, "custom")
    os.makedirs(base_job_custom, exist_ok=True)
    # recursive copy
    shutil.copytree(custom_data, base_job_custom, dirs_exist_ok=True)


    def prepare_nvflare_linear_experiment(
        site_naming_fn: callable,
        job_name: str, 
        site_prefix: str, 
        num_clients: int, 
        # split_method: str, 
        site_config_naming_fn: Callable[..., Any]
        # local_subsample: str, 
        # lr_mode: str
    ) -> dict:

        # make the folder for the job
        path = os.path.join(ALL_JOBS_PATH, job_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True) 

        job = {
            "prefix": site_prefix,
            "n_sites": num_clients,
            # "split_method": split_method,
            # "local_subsample": local_subsample, # hyper-parameter https://www.r-bloggers.com/2021/08/feature-subsampling-for-random-forest-regression/
            "lr_scale": 1, # investigate this param: https://github.com/NVIDIA/NVFlare/blob/e217679c4de035564a6ed9c2e2658197b0c8e701/examples/advanced/random_forest/utils/prepare_job_config.py#L93
            # "lr_mode": lr_mode, # or "scaled"
            "nthread": num_clients,
            "_name": job_name,
            "_path": path
        }

        # 1. define the meta file for this job by augmenting the base file and writing it to a new job folder
        meta_config = read_json(base_job_meta_path)
        meta_config["name"] = job_name
        meta_config["deploy_map"] = get_deploy_map(job["prefix"], job["n_sites"])
        meta_config["min_clients"] = job["n_sites"]
        write_json(meta_config, os.path.join(path, JobConstants.META_FILE))


        # 2. define the server & client configs for this job by augmenting the base file and writing it to a new job folder
        # 2.1.1 make job server config
        server_config = read_json(base_job_server_config_path)
        server_config["min_clients"] = job["n_sites"]
        server_config["workflows"][0]["args"]["min_clients"] = job["n_sites"]

        server_app_name = SERVER_VAL
        server_config_path = os.path.join(path, server_app_name, "config")
        if not os.path.exists(server_config_path):
            os.makedirs(server_config_path, exist_ok=True)
        write_json(server_config, os.path.join(server_config_path, JobConstants.SERVER_JOB_CONFIG))

        # 2.2 make job client config
        for site_idx in range(job["n_sites"]):
            
            client_app_name = CLIENT_VAL(job["prefix"], site_idx + 1)
            client_path = os.path.join(path, client_app_name)
            if not os.path.exists(client_path):
                os.makedirs(client_path, exist_ok=True)

            client_config_path = os.path.join(client_path, "config")
            if not os.path.exists(client_config_path):
                os.makedirs(client_config_path, exist_ok=True)

            # 2.2.1 update client config
            client_config = read_json(base_job_client_config_path)
        
            # path for json which defines site split
            data_split_name = os.path.join(current_experiment.experiment_path, site_config_naming_fn(site_idx))
            
            client_config["components"][0]["args"]["data_split_filename"] = data_split_name
            client_config["components"][0]["args"]["client_id"] = site_naming_fn(site_idx)
            
            # https://github.com/NVIDIA/NVFlare/blob/1433290c203bd23f34c29e11795ce592bc067888/examples/advanced/sklearn-linear/utils/prepare_job_config.py#L190C5-L194C80
            write_json(client_config, os.path.join(client_config_path, JobConstants.CLIENT_JOB_CONFIG))

            # 2.2.2 copy over client custom files
            client_custom_path = os.path.join(client_path, "custom")
            if not os.path.exists(client_custom_path):
                os.makedirs(client_custom_path, exist_ok=True)
            shutil.copytree(base_job_custom, client_custom_path, dirs_exist_ok=True)

        return job

    site_prefix = "site-"

    # 1. split the data for the experiment so each client has its own dataframe of stratified data
    #  = current_experiment.get_stratified_client_subsets(
    #     dataset=current_experiment.training_dataset,
    #     num_clients=num_clients,
    #     method=split_method
    # )

    training_dfs, validation_dfs = current_experiment.get_client_subsets(
        dataset=current_experiment.training_dataset,
        num_clients=num_clients,
        method=split_method,
        stratified=stratified
    )
    
    # import numpy as np
    # training_dfs = [t.apply(lambda x: np.zeros_like(x)) for t in training_dfs]
    # for t in training_dfs:
    #     t['PHENO'] = np.random.randint(0, 2, t['PHENO'].shape[0])
    
    # translate data frames into client splits & write the data
    training_data_paths = []
    validaton_data_paths = []
    for i, (training_subset, validation_subset) in enumerate(zip(training_dfs, validation_dfs)):
        
        training_subset_path = os.path.join(current_experiment.experiment_path, f'client_training_{i}_stratified.h5')
        training_subset.to_hdf(training_subset_path, key='df', mode='w')
        training_data_paths.append(training_subset_path)

        validation_subset_path = os.path.join(current_experiment.experiment_path, f'client_validation_{i}_stratified.h5')
        validation_subset.to_hdf(validation_subset_path, key='df', mode='w')
        validaton_data_paths.append(validation_subset_path)

    def site_naming_fn(site_index):
        """Used for naming files in the client data split json"""
        return f"{site_prefix}{site_index + 1}"

    def to_subset_id(site_index: int):
        """Name client data subsets in a human readable fashion. site_number is 1 indexed"""
        return f'{split_method}_sid_{site_index}_of_{num_clients}.json'
    
    filenames, client_jsons = current_experiment.nvflare_multi_site_split_json(
        data_source_path=training_data_paths, 
        validation_data_source_path=validaton_data_paths,
        site_naming_fn=site_naming_fn,
        site_config_naming_fn=to_subset_id,
    )

    for filename, client_json in zip(filenames, client_jsons):
        print(filename, client_json)
        write_json(client_json, os.path.join(current_experiment.experiment_path, filename))

    # 2. define the simulation job

    # hyper-parem defaults from tutorial
    local_subsample=1 # 0.8 in tutorial, we have smaller dataset
    lr_mode="uniform"

    def get_job_name(local_subsample, lr_mode):
        """
        The unique id for this experiment in the context of NVFlare
        Args:
            local_subsample: Local random forest subsample rate https://github.com/NVIDIA/NVFlare/blame/e217679c4de035564a6ed9c2e2658197b0c8e701/examples/advanced/random_forest/utils/prepare_job_config.py#L38
            lr_mode: Whether to use uniform or scaled shrinkage
        """
        local_subsample = int(local_subsample * 100)
        return f"linear_{site_prefix}_{num_clients}_sites_{local_subsample}_ls_{split_method}_sm_{lr_mode}_lr"

    job_name = get_job_name(local_subsample, lr_mode)

    job_config = prepare_nvflare_linear_experiment(
        site_naming_fn=site_naming_fn,
        job_name=job_name,
        site_prefix=site_prefix,
        num_clients=num_clients,
        # split_method=split_method,
        site_config_naming_fn=to_subset_id, # used for getting the config file for the sites
        # local_subsample=local_subsample
        # lr_mode=lr_mode
    )

    workspace_path = f"/tmp/nvflare/workspaces/{job_name}"
    
    simulator = SimulatorRunner(
        job_folder=job_config["_path"],
        workspace=workspace_path,
        n_clients=job_config["n_sites"],
        threads=job_config["n_sites"]
    )
    run_status = simulator.run()
    print("Simulator finished with run_status", run_status)

    model_path = os.path.join(workspace_path, "simulate_job", SERVER_VAL, BEST_MODEL_SAVE)
    
    return model_path
