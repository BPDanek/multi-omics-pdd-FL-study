import pandas as pd
import numpy as np
import collections
import os
from sklearn import metrics
import sys
sys.path.append(os.getcwd())
from federated_learning_multi_modality_ancestry.multi_modality_fl.utils.delongs import delong_roc_test

def statistical_analysis(results_directory: str, save_path: str):
    
    internal_fed_reconstructed_results = []
    internal_cent_reconstructed_results = []

    external_fed_reconstructed_results = []
    external_cent_reconstructed_results = []

    best_fold_internal_cent = collections.defaultdict(dict)
    best_fold_internal_fed = collections.defaultdict(dict)

    best_fold_external_cent = collections.defaultdict(dict)
    best_fold_external_fed = collections.defaultdict(dict)

    all_results_internal = collections.defaultdict(pd.DataFrame)
    all_results_external = collections.defaultdict(pd.DataFrame)

    for dirpath, dirname, filenames in list(os.walk(results_directory)):

        for filename in filenames:
            if ".csv" not in filename:
                continue
            p = os.path.join(dirpath, filename)
            df = pd.read_csv(p)
            
            # all_dfs[filenames] = df
            for (val_name, algorithm_name, split_method, num_clients), group in df.groupby(['val_name', 'algorithm_name', 'split_method', 'num_clients'], group_keys=True):
                if num_clients != 0 and num_clients != 2:
                    continue

                auc = metrics.roc_auc_score(group['y_true'], group['y_pred'])
                
                # we need the fold idx of the best performing algorithm in the task
                # the fold idx can be different for internal and external test sets
                if val_name == 'internal test':
                    if 'fed' in algorithm_name.lower():
                        if (algorithm_name not in best_fold_internal_fed or auc > best_fold_internal_fed[algorithm_name]['score']):
                            best_fold_internal_fed[algorithm_name]['score'] = auc
                            best_fold_internal_fed[algorithm_name]['fold'] = group['fold_idx'].iloc[0]
                            best_fold_internal_fed[algorithm_name]['algorithm_name'] = algorithm_name

                        internal_fed_reconstructed_results.append(group)

                    else:
                        if (algorithm_name not in best_fold_internal_cent or auc > best_fold_internal_cent[algorithm_name]['score']):
                            best_fold_internal_cent[algorithm_name]['score'] = auc
                            best_fold_internal_cent[algorithm_name]['fold'] = group['fold_idx'].iloc[0]
                            best_fold_internal_cent[algorithm_name]['algorithm_name'] = algorithm_name
                        
                        internal_cent_reconstructed_results.append(group)

                    # aggregate results
                    all_results_internal[group['fold_idx'].iloc[0]] = pd.concat([all_results_internal[group['fold_idx'].iloc[0]], group])

                elif val_name == 'external test':
                    if 'fed' in algorithm_name.lower():
                        if (algorithm_name not in best_fold_external_fed or auc > best_fold_external_fed[algorithm_name]['score']):
                            best_fold_external_fed[algorithm_name]['score'] = auc
                            best_fold_external_fed[algorithm_name]['fold'] = group['fold_idx'].iloc[0]
                            best_fold_external_fed[algorithm_name]['algorithm_name'] = algorithm_name
                        
                        external_fed_reconstructed_results.append(group)
                    else:
                        if (algorithm_name not in best_fold_external_cent or auc > best_fold_external_cent[algorithm_name]['score']):
                            best_fold_external_cent[algorithm_name]['score'] = auc
                            best_fold_external_cent[algorithm_name]['fold'] = group['fold_idx'].iloc[0]
                            best_fold_external_cent[algorithm_name]['algorithm_name'] = algorithm_name

                        external_cent_reconstructed_results.append(group)

                    # aggregate results
                    all_results_external[group['fold_idx'].iloc[0]] = pd.concat([all_results_external[group['fold_idx'].iloc[0]], group])

                else:
                    raise ValueError('val_name not recognized')
                

    def create_delongs_table(test_subset, best_folds_cent, best_folds_fed, save_path):
        """
        Compute a table of statistical significance between central and federated algorithms using Delong's test
        """

        def sort_name(name):
            if "μ" in name:
                loc = name.find("Classifier")
                return "avg" if "Avg" in name else "prox" + name[loc-3:]
            else:
                return name
                
        def statistical_signifcance_table(test_subset, best_folds_cent, best_folds_fed, save_path):
            """
            Compute a table of statistical significance between central and federated algorithms using Delong's test
            """

            cent_algs_to_evaluate = [
                'LogisticRegression',
                'SGDClassifier',
                'MLPClassifier_0.1',
                'XGBRFClassifier_10'
            ]
            
            fed_algs_to_evaluate = [
                'FedAvg SGDClassifier',
                'FedAvg XGBRFClassifier',
                'FedAvg LRClassifier',
                'FedAvg MLPClassifier',
                'FedProx μ = 0 LRClassifier',
                'FedProx μ = 0 MLPClassifier',
                'FedProx μ = 2 LRClassifier',
                'FedProx μ = 2 MLPClassifier'
            ]
            
            results = []
            for central_alg in sorted(cent_algs_to_evaluate, key=sort_name):

                row = { 'central_alg': central_alg }
                for fed_alg in sorted(fed_algs_to_evaluate, key=sort_name):

                    # get the optimal model for this algorithm
                    cent_dataset = test_subset[best_folds_cent[central_alg]['fold']]
                    fed_dataset = test_subset[best_folds_fed[fed_alg]['fold']]

                    # use the optimal model as the basis for the delong's evaluation
                    cent_results = cent_dataset[cent_dataset['algorithm_name'] == central_alg].iloc[:cent_dataset['num_samples'].unique()[0]]
                    fed_results = fed_dataset[fed_dataset['algorithm_name'] == fed_alg].iloc[:fed_dataset['num_samples'].unique()[0]]

                    assert cent_results['y_true'].astype(int).to_list() == fed_results['y_true'].astype(int).to_list()

                    p_value_log10 = delong_roc_test(cent_results['y_true'], cent_results['y_pred'], fed_results['y_pred'])
                    p_value = 10 ** p_value_log10[0][0] # unpack and convert to normal scale
                    
                    cent_auc = metrics.roc_auc_score(cent_results['y_true'], cent_results['y_pred'])
                    fed_auc = metrics.roc_auc_score(fed_results['y_true'], fed_results['y_pred'])

                    cell_str = 'greater' if cent_auc > fed_auc else 'lesser'
                    
                    if (p_value < 0.05):
                        print(central_alg, fed_alg, p_value)
                        cell_str += '*'
                    else:
                        print(central_alg, fed_alg, p_value, 'not significant')

                    row[fed_alg] = cell_str
                results.append(row)

            table = pd.DataFrame.from_records(results)
            table.to_csv(save_path)
            return table

        statistical_signifcance_table(test_subset, best_folds_cent, best_folds_fed, f"{save_path}/external_test_stat_sig.csv")

    create_delongs_table(all_results_external, best_fold_external_cent, best_fold_external_fed, save_path)