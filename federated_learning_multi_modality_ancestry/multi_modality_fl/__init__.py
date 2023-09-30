import os

ANALYSIS_PATH = os.getcwd()
"""The base of the repos used for conducting data analysis"""

def MAKE_EXPERIMENT_DATA_PATH(experiment_name_str: str):
    """
    A workspace for experiments to produce intermediate processing objects. Example: 
    `~/repo_base/experiment_data`
    """
    EXPERIMENT_DATA_PATH = os.path.join(ANALYSIS_PATH, 'experiments', experiment_name_str) 
    os.makedirs(EXPERIMENT_DATA_PATH, exist_ok=True)
    return EXPERIMENT_DATA_PATH