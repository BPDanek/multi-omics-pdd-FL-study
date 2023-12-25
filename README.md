# Federated Learning for multi-omics: a performance evaluation in Parkinson’s disease 
Federated Learning for multi-omics: a performance evaluation in Parkinson’s disease 

Benjamin Danek (1,2,3), Mary B. Makarious (4,5,6), Anant Dadu (2,3), Dan Vitale (2,3), Paul Suhwan Lee (2), Mike A Nalls (2,3,4), Jimeng Sun (1,7), Faraz Faghri (2,3,4,*)

1- Department of Computer Science, University of Illinois at Urbana-Champaign, Champaign, IL, 61820, USA

2- Center for Alzheimer's and Related Dementias (CARD), National Institute on Aging and National Institute of Neurological Disorders and Stroke, National Institutes of Health, Bethesda, MD, 20892, USA

3- DataTecnica, Washington, DC, 20037, USA

4- Laboratory of Neurogenetics, National Institute on Aging, National Institutes of Health, Bethesda, MD, 20892, USA

5 - Department of Clinical and Movement Neurosciences, UCL Queen Square Institute of Neurology, London, UK

6 - UCL Movement Disorders Centre, University College London, London, UK

7- Carle Illinois College of Medicine, University of Illinois at Urbana-Champaign, Champaign, IL, 61820, USA

*- Lead contact

## Prerequisites
* conda version 23.5.0
* operating system: Red Hat Enterprise Linux, CentOS, Ubuntu, MacOS
* 20GB of RAM, 8 core CPU

## Setup Data File Structure
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

## Install Experiment Packages and Dependencies
`conda env create -f environment.yml -n multi_omics_pdd_fl`
This code will install dependencies listed in environment.yml, and create a new `Python 3.10.12` environment.

To activate the environment, run `conda activate multi_omics_pdd_fl`. If you would like to exit the environment call `conda deactivate`.
Note that it is possible to have several conda environments active simultaneously, which may cause unexpected dependency conflicts.

You may also use your IDE (VS Code, Jupyter Lab, etc.) to select this conda environment.

You can tell if you have other conda environments active in the shell:
```
(base) (multi_omics_pdd_fl) benjamindanek@bd federated_learning_multi_modality_ancestry % 
```
Indicates the environment "base" is active, in addition to the environment (multi_omics_pdd_fl). If you enter the command `conda deactivate` twice, both environments will be popped. Then you can activate solely the one you want active. This "base" environment is the install default from conda, and tends to be active when a new shell window opens. 

## Run The Experiments  
To run the suite of experiments used to generate the paper results, run the following from the GitHub repository's parent directory: 
```
$ python federated_learning_multi_modality_ancestry/multi_modality_fl/experiment_runner/run_experiments.py -d <path to data source>
```

The shell should look like:
```
(multi_omics_pdd_fl) benjamindanek@bd federated_learning_multi_modality_ancestry/multi_modality_fl/experiment_runner/run_experiments.py -d <path to data source>
```
There will be a series of run logs which are outputted. These huge volume of logs is due to the experiment simulations `NVFlare` and `flower` output. 
To control the simulation outputs, one will need to set the logging configurations for those packages:
https://nvflare.readthedocs.io/en/2.3.0/user_guide/logging_configuration.html
https://flower.dev/docs/framework/how-to-configure-logging.html

### Configure The Experiments
To modify the experiment suite, adjust the series of experiments run in the `federated_learning_multi_modality_ancestry/multi_modality_fl/experiment_runner/run_experiments.py` file. This interface also allows settinghyper parameters for experiment runs.
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

## Visualizing Experiment Results
Experiment results are written to the directories:
```
federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_uniform_strat/
federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_uniform_non_strat/
federated_learning_multi_modality_ancestry/multi_modality_fl/results/manual_experiments_linear_non_strat/
```
Results are written for each `fold_idx`, and for each dataset (internal, external). For the full suite of experiments, all 6 folds, there should be 6 `.csv` files for each test set in each of the above directories.

To generate figures, run the notebook: `federated_learning_multi_modality_ancestry/multi_modality_fl/results/build_figures.ipynb`

Figures and tables will be outputted in the folder `federated_learning_multi_modality_ancestry/multi_modality_fl/results/generated_figures_tables`. A fully rendered run of the notebook is available [here](federated_learning_multi_modality_ancestry/multi_modality_fl/results/build_figures.ipynb)

## Features
The full list of features used in the Parkinson's disease classification task is available in the [features csv file](https://github.com/BPDanek/multi-omics-pdd-FL-study/blob/main/data/features.csv). Details on how features were generated, are referenced in the paper, and comprehensively explained in [1]

The top features, measured by feature importance per [1] (figure 4).

| Feature ID      | Feature Source     |
|:----------------|:-------------------|
| AGE             | Clinico-demographic|
| MALE            | Clinico-demographic|
| FAMILY_HISTORY  | Clinico-demographic|
| UPSIT           | Clinico-demographic|
| InfAJ           | Clinico-demographic|
| PRS90           | Genetic            |
| ENSG00000153976 | Transcriptomic     |
| rs10835060      | Genetic            |
| ENSG00000182447 | Transcriptomic     |
| ENSG00000132780 | Transcriptomic     |
| ENSG00000197591 | Transcriptomic     |
| ENSG00000140478 | Transcriptomic     |
| ENSG00000072609 | Transcriptomic     |
| ENSG00000101605 | Transcriptomic     |
| ENSG00000100079 | Transcriptomic     |
| ENSG00000189430 | Transcriptomic     |
| ENSG00000105792 | Transcriptomic     |
| ENSG00000180530 | Transcriptomic     |
| ENSG00000136560 | Transcriptomic     |
| ENSG00000204248 | Transcriptomic     |
| ENSG00000165806 | Transcriptomic     |
| ENSG00000184260 | Transcriptomic     |
| rs4238361       | Genetic            |
| ENSG00000162739 | Transcriptomic     |

## Tutorial
You can find tutorial notebooks that will guide you in reproducing our results. You can find them in the directory `notebooks/tutorial`

Inside the tutorial directory, you will find the following notebooks:
- 01_installation.ipynb
- 02_repo_structure.ipynb
- 03_running_fl.ipynb
- 04_visualizing_results.ipynb

References:
1. Makarious, Mary B., Hampton L. Leonard, Dan Vitale, Hirotaka Iwaki, Lana Sargent, Anant Dadu, Ivo Violich, et al. 2022. “Multi-Modality Machine Learning Predicting Parkinson’s Disease.” Npj Parkinson’s Disease 8 (1): 35.
