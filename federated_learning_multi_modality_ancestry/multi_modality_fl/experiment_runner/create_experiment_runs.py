# Generate the Cartesian product of the hyperparameters
import itertools

hyper_params = {
    "num_rounds": [5],
    "num_local_rounds": [300],# [1, 300, 500],#, 10],
    "client_lr": [1e-3],  #[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "site_configurations": [2, 4, 6, 8, 10, 12],
    "split_methods": ["uniform", "polynomial"],#, "polynomial"],
    "stratified": [True, False],
}

experiments_to_run = list(itertools.product(
    hyper_params['num_rounds'],
    hyper_params['num_local_rounds'],
    hyper_params['client_lr'],
    hyper_params['site_configurations'],
    hyper_params['split_methods'],
    hyper_params['stratified']
))

# Define the output file path
output_file = 'experiment_runs.txt'

# Write the results to the output file
with open(output_file, 'w') as file:
    for e in experiments_to_run:
        experiment = f"{e[0]} {e[1]} {e[2]} {e[3]} {e[4]} {e[5]}"
        file.write(f"{experiment}\n")

print(f"Results written to {output_file}")