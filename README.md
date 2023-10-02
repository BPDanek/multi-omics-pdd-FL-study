# FL-for-multi-omics-pdd
Federated Learning for multi-omics: a performance evaluation in Parkinson’s disease 

## prerequisites
* conda version 23.5.0
* operating system: MacOS Ventura
* 20GB of RAM, 8 core CPU

## setup data file structure
From the `DATASET_PATH` the assumed structure of the experiments is
```
.
├── Combined_G1E5_O1E2
│   └── PPMI-genetic_p1E5_omic_p1E2.dataForML.h5
├── Validation
    └── validate-PDBP-genetic_p1E5_omic_p1E2.dataForML.h5
```
These files can be produced by reproducing the processing approach from 
```
Makarious, M.B., Leonard, H.L., Vitale, D. et al. Multi-modality machine learning predicting Parkinson’s disease. npj Parkinsons Dis. 8, 35 (2022). https://doi.org/10.1038/s41531-022-00288-w
```
Available publicly at: https://github.com/GenoML/GenoML_multimodal_PD/

## install experiment packages and dependencies
`conda env create -f environment.yml -n multi_omics_pdd_fl`
This code will install dependencies listed in environment.yml, and create a new `Python 3.10.12` environment.

To activate the environment, run `conda activate multi_omics_pdd_fl`. If you would like to exit the environment call `conda deactivate`.
Note that it is possible to have several conda environments active simultaneously, which may cause unexpected dependency conflicts. 

You can tell if you have other conda environments active in the shell:
```
(base) (multi_omics_pdd_fl) benjamindanek@bd federated_learning_multi_modality_ancestry % 
```
Indicates the environment "base" is active, in addition to the environment (multi_omics_pdd_fl). If you enter the command `conda deactivate` twice, both environments will be popped. Then you can activate solely the one you want active. This "base" environment is the install default from conda, and tends to be active when a new shell window opens. 

## run the experiments
Before running experiments, the variables `REPO_PARENT` and `DATASET_PATH` must be updated to whatever values are meaningful for the test machine. Once these values have been set, the experiments can be run.
* `REPO_PARENT` is used to define the path of the folder which contains this gith repository.
* `DATASET_PATH` is used to define the folder which includes the dataasets used for experimentation.
  
To run the suite of experiments used to generate the paper results, run the following from the repository root: `python multi_modality_fl/experiment_runner/run_experiments.py`
The shell should look like:
```
(multi_omics_pdd_fl) benjamindanek@bd federated_learning_multi_modality_ancestry % python multi_modality_fl/experiment_runner/run_experiments.py
```
There will be a series of run logs which are outputted. These huge volume of logs is due to the experiment simulations `NVFlare` and `flower` output. 
To control the simulation outputs, one will need to set the logging configurations for those packages:
https://nvflare.readthedocs.io/en/2.3.0/user_guide/logging_configuration.html
https://flower.dev/docs/framework/how-to-configure-logging.html

###  configure the experiments
To modify the experiment suite, adjust the series of experiments run in the `multi_modality_fl/experiment_runner/run_experiments.py` file. This interface also allows settinghyper parameters for experiment runs.
The function 
```
run_baseline_exp
```
will run the suite of classical ML algorithms used as baseline performance.

The function
```
run_exp
```

runs several FL algorithms side-by-side. These algorithms share the same parameters:
|variable name | description |
| --- | --- |
|num_rounds| The number of FL training rounds|
|num_local_rounds| the number of local rounds to run the local learner for|
|client_lr| the learning rate at the client sites. Some local learner algorithms do not use learning rate as a parameter|
|site_config| the number of clients in the federation|
|split_method| the split method used for distributing clients (ie uniform, linear)|
|stratified| whether to use stratified sampling (true == stratified sampling, false == random sampling without stratification)|

## visualizing experiment results
Experiment results are written to the directories:
```
federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_uniform_strat/
federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_uniform_non_strat/
federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_linear_non_strat/
```
Results are written for each `fold_idx`, and for each dataset (internal, external). For the full suite of experiments, all 6 folds, there should be 6 `.csv` files for each test set in each of the above directories.

To generate figures, run the file:
```
python multi_modality_fl/results/global_experiment_plots.py
```
Figures and tables will be outputted in the folder `multi_modality_fl/results/generated_figures_tables`.
