
import os
from typing import List, Union
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"

colors = {
    'FedAvg LRClassifier': 'rosybrown',
    'FedProx μ = 0 LRClassifier': 'lightcoral', # 0.5
    'FedProx μ = 2 LRClassifier': 'indianred',
    'FedProx μ = 5 LRClassifier': 'firebrick',
    'FedProx μ = 8 LRClassifier': 'maroon',

    'FedAvg MLPClassifier': 'magenta',
    'FedProx μ = 0 MLPClassifier': 'orchid', # 0.5
    'FedProx μ = 2 MLPClassifier': 'deeppink',
    'FedProx μ = 5 MLPClassifier': 'hotpink',
    'FedProx μ = 8 MLPClassifier': 'pink',

    'FedAvg SGDClassifier': 'limegreen',

    'FedAvg XGBRFClassifier': 'deepskyblue',
}

def make_new_bar_chart(dataset: str, y_range: List, base_output_path: str, filename: str, df: pd.DataFrame, selected_num_clients: int, metric: str, show_error: bool, write: bool = True):

    # bar_group_names = [
    #     'LogisticRegression',
    #     'MLPClassifier',
    #     'SGDClassifier',
    #     'XGBRFClassifier',
    #     'GradientBoostingClassifier', # central only; best overall performance on central
    #     # 'AdaBoostClassifier', # central only; best overall performance on central
    # ]

    def make_score_dict():
        return {
            'score': 0,
            'error': 0,
            'color': 'orange'
        }

    barchart_performance_groups = {
        'LogisticRegression': {
            'LogisticRegression': make_score_dict(),
            'FedAvg LRClassifier': make_score_dict(),
            'FedProx μ = 0 LRClassifier': make_score_dict(),
            'FedProx μ = 2 LRClassifier': make_score_dict(),
            # 'FedProx μ = 5 LRClassifier': make_score_dict(),
            # 'FedProx μ = 8 LRClassifier': make_score_dict()
        },
        'MLPClassifier': {
            'MLPClassifier': make_score_dict(),
            'FedAvg MLPClassifier': make_score_dict(),
            'FedProx μ = 0 MLPClassifier': make_score_dict(),
            'FedProx μ = 2 MLPClassifier': make_score_dict(),
            # 'FedProx μ = 5 MLPClassifier': make_score_dict(),
            # 'FedProx μ = 8 MLPClassifier': make_score_dict()
        },
        'SGDClassifier': {
            'SGDClassifier': make_score_dict(),
            'FedAvg SGDClassifier': make_score_dict()
        },
        'XGBRFClassifier': {
            'XGBRFClassifier': make_score_dict(),
            'FedAvg XGBRFClassifier': make_score_dict()
        },
        # 'GradientBoostingClassifier': {
        #     'XGBRFClassifier': make_score_dict()
        # },
        # 'AdaBoostClassifier': [],
    }

    selected_cent_results = df[df['num_clients'] == 0]
    selected_cent_results.set_index('algorithm_name', inplace=True)

    selected_fed_results = df[df['num_clients'] == selected_num_clients]
    selected_fed_results.set_index('algorithm_name', inplace=True)

    for alg_group_name in barchart_performance_groups:
        for alg_name, alg_score_dict in barchart_performance_groups[alg_group_name].items():
            
            if alg_name in selected_fed_results.index:
                alg_score_dict['score'] = selected_fed_results.loc[alg_name, metric]
                alg_score_dict['error'] = selected_fed_results.loc[alg_name, f"{metric}_std"]
                if 'prox' in alg_name.lower():
                    if '0' in alg_name:
                        alg_score_dict['color'] = 'skyblue'
                    if '2' in alg_name:
                        alg_score_dict['color'] = 'dodgerblue'
                    if '5' in alg_name:
                        alg_score_dict['color'] = 'darkblue'
                    if '8' in alg_name:
                        alg_score_dict['color'] = 'midnightblue'
                else:
                    alg_score_dict['color'] = 'yellowgreen'
            else:
                try:
                    alg_score_dict['score'] = selected_cent_results.loc[alg_name, metric]
                    alg_score_dict['error'] = selected_cent_results.loc[alg_name, f"{metric}_std"]
                    alg_score_dict['color'] = 'lightsalmon'
                except KeyError:
                    continue

    print(barchart_performance_groups)

    num_groups = len(barchart_performance_groups)
    
    # Set the width of each bar within a group
    bar_width = 0.5
    gap_between_groups = 0.1

    # Create a list of x positions for each group
    x_positions = range(num_groups)

    # Create a figure and an Axes object
    fig, ax = plt.subplots()

    # Iterate through each group and plot the bars with custom colors
    # for i, group_name in enumerate(barchart_performance_groups):
    #     x = [pos + i * bar_width for pos in x_positions]
    #     bars = ax.bar(x, bar_values[i], bar_width, label=group_name, color=bar_colors[i])

    x_ticks = []
    loc = 0.5
    for i, (group_name, group_dict) in enumerate(barchart_performance_groups.items()):

        group_start = loc
        for j, alg_name in enumerate(group_dict):
            
            score = group_dict[alg_name]['score']
            error = group_dict[alg_name]['error']
            color = group_dict[alg_name]['color']

            alg_label = 'Central'
            if 'FedAvg' in alg_name:
                alg_label = f'Federated (FedAvg, n={selected_num_clients})'
            if 'FedProx μ = 0' in alg_name:
                alg_label = f'Federated (FedProx, μ = 0.5 n={selected_num_clients})'
            if 'FedProx μ = 2' in alg_name:
                alg_label = f'Federated (FedProx, μ = 2 n={selected_num_clients})'
            if 'FedProx μ = 5' in alg_name:
                alg_label = f'Federated (FedProx, μ = 5 n={selected_num_clients})'
            if 'FedProx μ = 8' in alg_name:
                alg_label = f'Federated (FedProx, μ = 8 n={selected_num_clients})'
            else:
                print("cent")


            if i == 0:
                ax.bar(loc, score, bar_width, label=alg_label, color=color, yerr=error, align='edge')
            else:
                # each label will be duplicated in the legend if we add it to each bar
                ax.bar(loc, score, bar_width, color=color, yerr=error, align='edge')

            loc += bar_width + gap_between_groups
        
        group_end = loc - gap_between_groups
        group_center = (group_start + group_end) / 2
        x_ticks.append(group_center)

        loc += 0.5

    # Customize the plot
    ax.set_xlabel(f'Algorithms')
    ax.set_ylabel('AUC Precision Recall')
    ax.set_title(f'Central vs. Federated Performance ({dataset})')
    ax.set_xticks(x_ticks)
    # x_tick_positions = np.arange(num_groups) + (num_groups - 1) * (bar_width + gap_between_groups) / 2
    # ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(barchart_performance_groups.keys(), fontsize=10, rotation=20, ha="right", rotation_mode="anchor")
    ax.grid(axis='y')
    ax.set_ylim(y_range[0], y_range[1])
    ax.legend(ncols=1, loc='lower right', framealpha=1)
    # ax.legend()

    # Show the plot
    plt.tight_layout()
    # plt.show()

    name = f"global_experiments_{filename}_compare_bar_chart.png"
    if write:
        plt.savefig(os.path.join(base_output_path, name))


def make_client_line_plot(dataset: str, y_range: List, base_output_path: str, filename: str, df: pd.DataFrame, metric: str, show_error: bool, write: bool = False):

    # get relevant data
    overall_table = df[df['split_method'] != 'central']

    algorithms = overall_table['algorithm_name'].unique()
    
    # make the table
    table_figure = overall_table.copy()
    # results_df['algorithm_name'] = results_df['algorithm_name'].apply(lambda x: x.replace('FedProx μ = 0', 'FedProx μ = 0.5') if type(x) == str else x)
    table_figure['algorithm_name'] = table_figure['algorithm_name'].apply(lambda x: x.replace('Classifier', '') if type(x) == str else x)
    table_figure = table_figure.round(3)
    table_figure[metric] = table_figure[metric].astype(str) + ' ± ' + table_figure[f"{metric}_std"].astype(str)

    table_figure = table_figure.set_index(['algorithm_name']).pivot(columns='num_clients')[metric]
    table_figure_path = os.path.join(base_output_path, f"global_experiments_{filename}_client_line_plot_table.csv")
    table_figure.to_csv(table_figure_path)

    # Create the scatter plot
    plt.figure()

    for alg in algorithms:

        df = overall_table[overall_table['algorithm_name'] == alg]
        
        x_values = df['num_clients']
        y_values = df[metric]

        error = df[f"{metric}_std"]
        y_upper = y_values + error
        y_lower = y_values - error

        
        if show_error:
            # Plot the error bands
            plt.fill_between(x_values, y_lower, y_upper, alpha=0.1, color=colors[alg])

        # Plot the main scatter plot
        plt.plot(x_values, y_values, label=alg.replace("FedProx μ = 0", "FedProx μ = 0.5") if "FedProx μ = 0" in alg else alg, color=colors[alg])

    # visualization improvements
    plt.ylim(y_range[0], y_range[1])
    # plt.xlim([2, 12])
    plt.xlim([2, 18])
    plt.grid(True)

    # Set labels and title
    plt.xlabel("Number of Client Sites")
    plt.ylabel("AUC Precision Recall")
    plt.title(f"FL Algorithm Performance vs. Client Sites ({dataset})", pad=25)

    # Show legend
    # plt.legend(ncol=2, bbox_to_anchor=(0.60, 0.35))
    plt.legend(ncol=2, loc='lower right')

    name = f"global_experiments_{filename}_client_line_plot.png"
    if write:
        plt.savefig(os.path.join(base_output_path, name))
    
    # plt.show()

    return name, plt


def generate_plots(dataset, y_range, results_directory: str, filename: Union[str, List[str]], file_extension: str):
    # write all figures and tables to this folder
    base_output_path = '/Users/benjamindanek/Code/federated_learning_multi_modality_ancestry/multi_modality_fl/results/generated_figures_tables'

    if not isinstance(filename, str):
        results_df = pd.DataFrame()
        for _filename in filename:
            data_path = os.path.join(results_directory, f"{_filename}.{file_extension}")
            current_res = pd.read_csv(data_path)
            results_df = pd.concat([results_df, current_res]).reset_index(drop=True)

        filename = f'k_all {dataset} test'
    else:
        data_path = os.path.join(results_directory, f"{filename}.{file_extension}")
        results_df = pd.read_csv(data_path)
    
    results_df.drop(columns=['num_samples'], inplace=True)
    
    results_df = results_df[results_df['algorithm_name'] != 'BaggingClassifier']

    results_df = results_df[results_df['algorithm_name'] != 'FedProx μ = 5 LRClassifier']
    results_df = results_df[results_df['algorithm_name'] != 'FedProx μ = 5 MLPClassifier']

    results_df = results_df[results_df['algorithm_name'] != 'FedProx μ = 8 LRClassifier']
    results_df = results_df[results_df['algorithm_name'] != 'FedProx μ = 8 MLPClassifier']

    results_df['algorithm_name'] = results_df['algorithm_name'].apply(lambda x: x.replace('_0.1', '') if type(x) == str else x)
    results_df['algorithm_name'] = results_df['algorithm_name'].apply(lambda x: x.replace('_10', '') if type(x) == str else x)
    # results_df['algorithm_name'] = results_df['algorithm_name'].apply(lambda x: x.replace('FedProx μ = 0', 'FedProx μ = 0.5') if type(x) == str else x)

    def computeAUCPR(y_true, y_pred):
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
        return metrics.auc(recall, precision)

    def _compute_metrics(group):
        _metrics = {}
        
        _metrics['roc_auc_score'] = metrics.roc_auc_score(group['y_true'], group['y_pred'])
        _metrics['auc_precision_recall'] = computeAUCPR(group['y_true'], group['y_pred'])

        # accuracy score
        _metrics['balanced_accuracy_score'] = metrics.balanced_accuracy_score(group['y_true'], group['y_pred'] >= 0.5)

        # calculate roc curves
        precision, recall, thresholds = metrics.precision_recall_curve(group['y_true'], group['y_pred'])

        # precision
        for i in np.argsort(precision):
            if i < len(thresholds):
                precision_optinal_thresh = thresholds[i]
                break

        _metrics['precision_score'] = metrics.precision_score(group['y_true'], group['y_pred'] >= precision_optinal_thresh)

        # recall
        recall_optinal_thresh = thresholds[np.argmax(recall)]
        _metrics['recall_score'] = metrics.recall_score(group['y_true'], group['y_pred'] >= recall_optinal_thresh)



        f_beta = lambda beta, precision, recall: ((1 + beta**2) * (precision * recall)) / ((beta**2) * precision + recall)
        
        # f0.5
        f_beta_0_5 = f_beta(0.5, precision, recall)
        f05_optimal_thresh = thresholds[np.argmax(f_beta_0_5)]
        _metrics['fbeta_score_0.5'] = metrics.fbeta_score(group['y_true'], group['y_pred'] >= f05_optimal_thresh, beta=0.5)
        
        # f1
        f_beta_1 = f_beta(1, precision, recall)
        f1_optimal_thresh = thresholds[np.argmax(f_beta_1)]
        _metrics['fbeta_score_1'] = metrics.fbeta_score(group['y_true'], group['y_pred'] >= f1_optimal_thresh, beta=1)

        # f2
        f_beta_2 = f_beta(2, precision, recall)
        f2_optimal_thresh = thresholds[np.argmax(f_beta_2)]
        _metrics['fbeta_score_2'] = metrics.fbeta_score(group['y_true'], group['y_pred'] >= f2_optimal_thresh, beta=2)

        # log loss
        _metrics['log_loss'] = metrics.log_loss(group['y_true'], group['y_pred'])

        # MCC
        _metrics['matthews_corrcoef'] = metrics.matthews_corrcoef(group['y_true'], group['y_pred'] >= 0.5)

        return _metrics


    groups = results_df.groupby(['val_name', 'algorithm_name', 'split_method', 'num_clients', 'fold_idx'], group_keys=True)
    reconstructed_results = []
    for (val_name, algorithm_name, split_method, num_clients, fold_idx), group in groups:
        
        # if num_clients > 12 and algorithm_name in ['LogisticRegression', 'FedAvg LR', 'FedProx 5.0 LR', 'FedProx 8.0 LR']:
        #     continue

        
        _row = {}
        _row['val_name'] = val_name
        _row['algorithm_name'] = algorithm_name
        _row['split_method'] = split_method
        _row['num_clients'] = num_clients
        _row['fold_idx'] = fold_idx
        
        _metrics = _compute_metrics(group)
        _row.update(_metrics)
        reconstructed_results.append(_row)

        print(algorithm_name, _row)

    reconstructed_results_df = pd.DataFrame(reconstructed_results)
    write_path = os.path.join(base_output_path, f"raw_results_table_{filename}.csv")
    reconstructed_results_df.to_csv(write_path, index=False)
    print("Wrote reconstructed_results_df to: ", write_path)

    mean_across_kfolds = reconstructed_results_df.groupby(['algorithm_name', 'num_clients', 'split_method', 'val_name']).mean().reset_index()
    std_across_kfolds = reconstructed_results_df.groupby(['algorithm_name', 'num_clients', 'split_method', 'val_name']).std().reset_index()
    std_across_kfolds = std_across_kfolds.add_suffix('_std')

    mean_std_reconstructed_metrics_table = pd.concat([mean_across_kfolds, std_across_kfolds], axis=1)
    write_path = os.path.join(base_output_path, f"k_averaged_results_table_{filename}.csv")
    mean_std_reconstructed_metrics_table.to_csv(write_path, index=False)
    print("Wrote mean_std_reconstructed_metrics_table to: ", write_path)

    # make_bar_chart(base_output_path, filename, mean_std_reconstructed_metrics_table, 2, metric='precision_score', show_error=True, write=True)
    make_new_bar_chart(dataset, y_range, base_output_path, filename, mean_std_reconstructed_metrics_table, 2, metric='auc_precision_recall', show_error=True, write=True)
    make_client_line_plot(dataset, y_range, base_output_path, filename, mean_std_reconstructed_metrics_table, metric='auc_precision_recall', show_error=True, write=True)


if __name__ == "__main__":
    
    results_base = '/Users/benjamindanek/Downloads/cell_patterns_submission_materials/federated_learning_multi_modality_ancestry/multi_modality_fl/results'

    # the number of k folds you ran the experiment for
    max_k = 1
    external_results_filename = [f'global_experiment_runner_k{i}_external test' for i in range(0, max_k)]
    internal_results_filenames = [f'global_experiment_runner_k{i}_internal test' for i in range(0, max_k)]

    results_directory = os.path.join(results_base, 'manual_experiments_uniform_strat')
    mixin = '/Users/benjamindanek/Downloads/nih_fl_pred_proba/red_fedlogreg_uniform_strat'    
    generate_plots("PDBP Uniform Stratified", [0.7, 0.9], results_directory, external_results_filename, 'csv')
    generate_plots("PPMI Uniform Stratified", [0.725, 0.975], results_directory, internal_results_filenames, 'csv')

    # results_directory = os.path.join(results_base, 'manual_experiments_uniform_non_strat')
    # generate_plots("PDBP Uniform Random", [0, 1], results_directory, external_results_filename, 'csv')
    # generate_plots("PPMI Uniform Random", [0, 1], results_directory, internal_results_filenames, 'csv')

    # results_directory = os.path.join(results_base, 'manual_experiments_linear_non_strat')
    # generate_plots("PDBP Linear Random", [0, 1], results_directory, external_results_filename, 'csv')
    # generate_plots("PPMI Linear Random", [0, 1], results_directory, internal_results_filenames, 'csv')
