import os
import sys
from pathlib import Path

REPO_PARENT = Path(__file__).parents[3]

sys.path.append(os.path.abspath("."))
sys.path.append(str(REPO_PARENT))

from math import ceil
import pandas as pd
import numpy as np
import time
import xgboost
from sklearn import discriminant_analysis, ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
import collections

from federated_learning_multi_modality_ancestry.multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration, write_json, read_json

hyper_param_logs_internal = []
hyper_param_logs_external = []

def run_baseline_experiments(current_experiment: GlobalExperimentsConfiguration, fold_idx: int):
    # utils
    def utils_time_fn(fun, *args, **kwargs):
        """return (function run time (second), result of function call)"""
        start_time = time.perf_counter()
        
        result = fun(*args, **kwargs)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        return (run_time, result)

    def utils_sk_metrics_to_str(metrics_dict):
        """convert metrics dict to readable object"""
        rows = []
        for key, value in metrics_dict.items():
            if key == "algorithm":
                rows.append("{}: {}".format(key, value))
            elif key == "runtime_s":
                rows.append("{}: {:0.3f} seconds\n".format(key, value))
            else:
                rows.append("{}: {:0.4f}".format(key, value))
        return str.join("\n", rows)
    
    # extracted GenoML
    candidate_algorithms = [
        linear_model.LogisticRegression(solver='lbfgs', random_state=current_experiment.RANDOM_SEED),
        ensemble.RandomForestClassifier(n_estimators=100, random_state=current_experiment.RANDOM_SEED),
        ensemble.AdaBoostClassifier(random_state=current_experiment.RANDOM_SEED),
        ensemble.GradientBoostingClassifier(random_state=current_experiment.RANDOM_SEED),
        linear_model.SGDClassifier(loss='log_loss', learning_rate='optimal', early_stopping=True, fit_intercept=1, random_state=current_experiment.RANDOM_SEED),
        # (linear_model.SGDClassifier(loss='log_loss', learning_rate='optimal', eta0=1, early_stopping=True, fit_intercept=1, random_state=current_experiment.RANDOM_SEED), 1),
        # (linear_model.SGDClassifier(loss='log_loss', learning_rate='optimal', eta0=10, early_stopping=True, fit_intercept=1, random_state=current_experiment.RANDOM_SEED), 10),
        # (linear_model.SGDClassifier(loss='log_loss', learning_rate='optimal', eta0=100, early_stopping=True, fit_intercept=1, random_state=current_experiment.RANDOM_SEED), 100),
        svm.SVC(probability=True, gamma='scale', random_state=current_experiment.RANDOM_SEED),
        # (neural_network.MLPClassifier(random_state=current_experiment.RANDOM_SEED), 0.001),
        (neural_network.MLPClassifier(learning_rate_init=0.1, random_state=current_experiment.RANDOM_SEED), 0.1),
        # (neural_network.MLPClassifier(learning_rate_init=1, random_state=current_experiment.RANDOM_SEED), 1),
        # (neural_network.MLPClassifier(learning_rate_init=10, random_state=current_experiment.RANDOM_SEED), 10),
        # (neural_network.MLPClassifier(learning_rate_init=0.01, random_state=current_experiment.RANDOM_SEED), 0.01), 
        # (neural_network.MLPClassifier(learning_rate_init=0.0001, random_state=current_experiment.RANDOM_SEED), 0.0001),
        neighbors.KNeighborsClassifier(),
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
        ensemble.BaggingClassifier(random_state=current_experiment.RANDOM_SEED),
        xgboost.XGBClassifier(random_state=current_experiment.RANDOM_SEED),
        # (xgboost.XGBRFClassifier(random_state=current_experiment.RANDOM_SEED), 1),
        (xgboost.XGBRFClassifier(learning_rate=10, random_state=current_experiment.RANDOM_SEED), 10),
        # (xgboost.XGBRFClassifier(learning_rate=100, random_state=current_experiment.RANDOM_SEED), 100)
    ]

    algorithms = {}
    for algorithm in candidate_algorithms:
        if type(algorithm) == tuple:
            algorithms[algorithm[0].__class__.__name__ + "_" + str(algorithm[1])] = algorithm[0]
        
        else:
            algorithms[algorithm.__class__.__name__] = algorithm
    
    print("\n".join(algorithms.keys()))

    def evaluate(competing_metrics, algorithm_name, algorithm, x, y):
        """evaluate how an algorithm does on the provided dataset & generate a pd row"""
        run_time, pred = utils_time_fn(algorithm.predict, x)
        metric_results = [metric_func(y, pred) for metric_func in competing_metrics]
        
        pred = algorithm.predict_proba(x)[:, 1]
        # m = metrics.precision_recall_curve(y, pred[:, 1])
        row = [algorithm_name, run_time] + metric_results # + [TN, FN, TP, FP, sensitivity, specificity, PPV, NPV]

        precision, recall, _ = metrics.precision_recall_curve(y, pred)
        aucpr = metrics.auc(recall, precision)
        return row, pred, aucpr


    def process_results(column_names, results):
        log_table = pd.DataFrame(data=results, columns=column_names)
        best_id = log_table.explained_variance_score.idxmax()
        best_algorithm_name = log_table.iloc[best_id].algorithm
        best_algorithm = algorithms[best_algorithm_name]
        best_algorithm_metrics = log_table.iloc[best_id].to_dict()
        
        res = {
            'log_table': log_table,
            'best_id': best_id,
            'best_algorithm_name': best_algorithm_name,
            'best_algorithm': best_algorithm,
            'best_algorithm_metrics': best_algorithm_metrics,
        }
        
        return res

    def compete(algorithms, x_train, y_train, x_test, y_test, x_addit_test=None, y_addit_test=None):
        """Compete the algorithms"""
        competing_metrics = [metrics.explained_variance_score, metrics.mean_squared_error,
                            metrics.median_absolute_error, metrics.r2_score, metrics.roc_auc_score,
                            metrics.average_precision_score]


        column_names = ["algorithm", "runtime_s"] + [metric.__name__ for metric in competing_metrics] # + ['TN', 'FN', 'TP', 'FP', 'sensitivity', 'specificity', 'PPV', 'NPV']

        results = []
        results_val = []
        for algorithm_name, algorithm in algorithms.items():

            algorithm.fit(x_train, y_train)
            
            row, y_pred, aucpr = evaluate(competing_metrics, algorithm_name, algorithm, x_test, y_test)
            results.append(row)
            current_experiment.log_raw_experiment_results(fold_idx=fold_idx, algorithm_name=algorithm_name, num_clients=0, split_method='central', stratified=False, val_name='internal test', num_rounds=-1, num_local_rounds=-1, client_lr=-1, proximal_mu=-1, y_true=y_test, y_pred=y_pred)
            hyper_param_logs_internal.append((fold_idx, algorithm_name, aucpr))

            row, y_addit_pred, aucpr_addit = evaluate(competing_metrics, algorithm_name, algorithm, x_addit_test, y_addit_test)
            results_val.append(row)
            current_experiment.log_raw_experiment_results(fold_idx=fold_idx, algorithm_name=algorithm_name, num_clients=0, split_method='central', stratified=False, val_name='external test', num_rounds=-1, num_local_rounds=-1, client_lr=-1, proximal_mu=-1, y_true=y_addit_test, y_pred=y_addit_pred)
            hyper_param_logs_external.append((fold_idx, algorithm_name, aucpr_addit))

        res = process_results(column_names, results)
        results_val = process_results(column_names, results_val)
        
        return res, results_val

    # def get_split(dataset, splits):
    #     indeces = list(range(0, len(dataset)))
    #     np.random.shuffle(indeces)
    #     subsets = []
    #     for portion in splits:
    #         offset = 0
    #         if (subsets):
    #             offset = len(subsets[-1])

    #         indeces_partition = ceil(len(dataset) * portion)
    #         subset = dataset[offset: min(offset + indeces_partition, len(dataset))]
    #         subset = subset.reset_index(drop=True)
    #         # print(f"split {portion} - len: {len(subset)} actual: {len(subset) / len(dataset)}")
    #         # display(subset)
    #         subsets.append(subset)

    #     return subsets
    
    # kfold_results = collections.defaultdict(lambda: collections.defaultdict(list))
    # for fold_idx in range(current_experiment.K):
    #     current_experiment.set_fold(fold_idx=fold_idx)

    # get processed datasets
    train = current_experiment.training_dataset
    internal, external = current_experiment.get_combined_test_dataset()
    test = internal[1]
    addit_test = external[1]
    
    # separate predictors
    x_train, y_train = current_experiment.as_features_labels(train, current_experiment.LABEL_COL)
    x_test, y_test = current_experiment.as_features_labels(test, current_experiment.LABEL_COL)
    x_external, y_external = current_experiment.as_features_labels(addit_test, current_experiment.LABEL_COL)
    

    result, result_val = compete(algorithms, x_train, y_train, x_test, y_test, x_external, y_external)
    return hyper_param_logs_internal, hyper_param_logs_external
    # print(result, result_val)

# if __name__ == "__main__":

#     current_experiment = GlobalExperimentsConfiguration(
#         base_path=os.path.join(os.getcwd(), os.path.join('multi_modality_fl', 'experiments')),
#         experiment_name='global_experiment_runner',
#         random_seed=0
#     )

#     current_experiment.create_experiment(
#         dataset_folder='/Users/benjamindanek/Code/federated_learning_multi_modality_ancestry/data',
#         dataset=GlobalExperimentsConfiguration.MULTIMODALITY,
#         split_method=GlobalExperimentsConfiguration.SKLEARN
#     )

#     print("Running baseline experiments")
    
#     current_experiment.set_fold(fold_idx=0)
    
#     run_baseline_experiments(current_experiment)
    
#     # print("Done.")    