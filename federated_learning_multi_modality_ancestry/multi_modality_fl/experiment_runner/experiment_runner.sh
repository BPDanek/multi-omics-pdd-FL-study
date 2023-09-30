#!/bin/bash

repo_parent="/home/danekbp"
write_results_path="../federated_learning_multi_modality_ancestry/multi_modality_fl/results/parallel_experiments/"
mkdir -p $write_results_path

# Define the input file pat
input_file="experiment_runs.txt"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file '$input_file' not found."
    exit 1
fi

# # run baselines
# for k in {0...5}
# do
#     python multi_modality_fl/experiment_runner/run_baseline_experiments.py $i $repo_parent $write_results_path
# done


run_python_command_with_retry() {
    
    # Command to run the Python program
    local python_command="$1"

    shift

    # echo $python_command

    # Number of retry attempts
    local max_retries=3

    # Delay between retries (in seconds)
    local retry_delay=5

    # Initialize retry counter
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        # Run the Python program
        $python_command

        # Check the exit status of the Python program
        if [ $? -eq 0 ]; then
            # If the program succeeded, exit the loop
            break
        else
            # If the program failed, increment the retry count and wait before retrying
            retry_count=$((retry_count + 1))
            echo "Python program failed (Retry $retry_count/$max_retries). Retrying in $retry_delay seconds... ($python_command))"
            sleep $retry_delay
        fi
    done

    # Check if the maximum number of retries has been reached
    if [ $retry_count -eq $max_retries ]; then
        echo "Max retries reached. Exiting."
    fi
}

for i in {0..5}
do
    while IFS= read -r line; do
        sleep 1
        
        echo "Processing experiment: $line"
        
        # You can perform further processing on each line here
        # For example, you can run a command using the values in the line
        run_python_command_with_retry "python multi_modality_fl/experiment_runner/run_single_experiment.py $i $line $repo_parent $write_results_path" &
    done < "$input_file"
done