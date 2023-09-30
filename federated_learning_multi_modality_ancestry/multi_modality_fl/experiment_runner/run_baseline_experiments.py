import os
import sys

import logging
logging.basicConfig(level=logging.ERROR)

sys.path.append(os.path.abspath('.'))

from multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration
from multi_modality_fl.experiments_code.baseline import run_baseline_experiments

# Check if command-line arguments are provided
if len(sys.argv) < 2:
    print("Usage: python print_args.py [arg1] [arg2] ...")
    sys.exit(1)

# Access and print command-line arguments
args = sys.argv[1:]

print("runnning baseline for k fold", args[0], " args:", args)

fold_idx = int(args[0])
repo_parent = args[1]

# experiments are deterministic
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

current_experiment.set_fold(fold_idx=fold_idx)

# performs logging
run_baseline_experiments(current_experiment, fold_idx) # learning rate for sgd?

write_paths = current_experiment.write_raw_experiment_results_as_df(fold_idx, f'{repo_parent}/federated_learning_multi_modality_ancestry/multi_modality_fl/results/global_experiment_results')
print(write_paths)
sys.exit(10)