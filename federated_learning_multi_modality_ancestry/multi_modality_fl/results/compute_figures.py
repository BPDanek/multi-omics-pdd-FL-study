
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


def generate_plots(base_output_path, dataset, y_range, results_directory: str, filename: Union[str, List[str]], file_extension: str, mixin=None):
    # write all figures and tables to this folder

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

        # precision
        _metrics['precision_score'] = metrics.precision_score(group['y_true'], group['y_pred'] >= 0.5)

        # recall
        _metrics['recall_score'] = metrics.recall_score(group['y_true'], group['y_pred'] >= 0.5)

        # calculate roc curves
        precision, recall, thresholds = metrics.precision_recall_curve(group['y_true'], group['y_pred'])

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


def create_tables(dataset, overall_table_path: str, output_path: str):
    
    data = pd.read_csv(overall_table_path)
    
    # get the appropriate table subsets
    baseline_selected = data[data['num_clients'] == 0]
    baseline_selected['Dataset'] = 'PPMI'

    fed_selected = data[data['num_clients'] == 2]
    fed_selected['Dataset'] = 'PPMI'

    combined = pd.concat([baseline_selected, fed_selected])

    combined['algorithm_name'] = combined['algorithm_name'].apply(lambda x: x.replace('Classifier', '') if type(x) == str else x)
    combined_clean = combined.set_index(['Dataset', 'algorithm_name']).drop(columns=['num_clients', 'split_method', 'val_name', 'fold_idx', 'num_clients_std', 'split_method_std', 'val_name_std', 'fold_idx_std', 'algorithm_name_std'])
    combined_clean = combined_clean.round(3)
    
    print("=====================================")
    print("Score table of all algorithms for:", dataset)    
    print(combined_clean.to_markdown())

    vanilla = []
    for c in combined_clean.columns:
        if '_std' not in c:
            vanilla.append(c)
        

    for v in vanilla:
        c = v + '_std'
        combined_clean[v] = combined_clean[v].astype(str) + '±' + combined_clean[c].astype(str)
        combined_clean = combined_clean.drop(columns=[c])

    combined_clean.to_csv(output_path)


def make_heterogeniety_figure(dataset: str, tables_path: str,num_clients: int = 2, metric: str = 'auc_precision_recall'):
    """
    Generates the heterogeniety figure for the paper.
    """
    assert dataset in ['PPMI', 'PDBP']

    # determine which dataset to generate the figure for (options: "PPMI" or "PDBP" for internal and external test set respectively)
    # dataset = "PDBP"
    NC = num_clients

    us_table = (f'k_averaged_results_table_k_all {dataset} Uniform Stratified test.csv', True) # uniform stratified
    ur_table = (f'k_averaged_results_table_k_all {dataset} Uniform Random test.csv', False) # uniform random
    lr_table = (f'k_averaged_results_table_k_all {dataset} Linear Random test.csv', False) # linear random

    tables = []
    for table_file_name, stratified in [us_table, ur_table, lr_table]: 
        table = pd.read_csv(os.path.join(tables_path, table_file_name))
        table = pd.concat([table[table['num_clients'] == 2], table[table['num_clients'] == 4]])
        table['algorithm_name'] = table['algorithm_name'].apply(lambda x: x.replace('Classifier', '') if type(x) == str else x).apply(lambda x: x.replace('FedProx μ = 0', 'FedProx μ = 0.5'))
        
        table['stratified'] = stratified
        tables.append(table)

    tables = pd.concat(tables)

    table_configs = {}
    table_subset = tables[tables['num_clients'] == NC]
    for (stratified, split_method), df in table_subset.drop(columns=['val_name', 'fold_idx', 'val_name_std', 'fold_idx_std', 'num_clients', 'num_clients_std']).groupby(['stratified', 'split_method']):
        print(stratified, split_method)
        df = df.set_index(['algorithm_name']).drop(columns=['algorithm_name_std', 'stratified', 'split_method', 'split_method_std'])
        # display(df)
        table_configs[(stratified, split_method)] = df

    uniform_stratified = table_configs[(True, 'uniform')]
    uniform_random = table_configs[(False, 'uniform')]
    linear_random = table_configs[(False, 'linear')]

    colors = [
        'gray',
        'salmon',
        'red',
    ]


    bar_width = 0.35
    group_centers = []

    f, ax = plt.subplots()
    f.set_figheight(4)
    f.set_figwidth(10)

    print(metric)

    t = tables[tables['num_clients'] == 2]
    std_ = f"{metric}_std"

    loc = 0

    label = True
    algorithm_names = ['FedAvg LR', 'FedProx μ = 0.5 LR', 'FedProx μ = 2 LR', 'FedAvg MLP', 'FedProx μ = 0.5 MLP', 'FedProx μ = 2 MLP', 'FedAvg SGD', 'FedAvg XGBRF']
    for alg_name in algorithm_names:
        
        if label:
            label = False
            loc += 0.5
            score = uniform_stratified.loc[alg_name][metric]
            error = np.abs(uniform_stratified.loc[alg_name][std_])
            ax.bar(loc, score, bar_width, label='Uniform Stratified', color=colors[0], yerr=error, align='edge')

            loc += 0.5
            group_centers.append(loc)
            score = uniform_random.loc[alg_name][metric]
            error = uniform_random.loc[alg_name][std_]
            ax.bar(loc, score, bar_width, label='Uniform Random', color=colors[1], yerr=error, align='edge')

            loc += 0.5
            score = linear_random.loc[alg_name][metric]
            error = linear_random.loc[alg_name][std_]
            ax.bar(loc, score, bar_width, label='Linear Random', color=colors[2], yerr=error, align='edge')
        else:
            loc += 0.5
            score = uniform_stratified.loc[alg_name][metric]
            error = np.abs(uniform_stratified.loc[alg_name][std_])
            ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

            loc += 0.5
            group_centers.append(loc + 0.25)
            score = uniform_random.loc[alg_name][metric]
            error = uniform_random.loc[alg_name][std_]
            ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

            loc += 0.5
            score = linear_random.loc[alg_name][metric]
            error = linear_random.loc[alg_name][std_]
            ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')

        loc += 0.5

    ax.set_xticks(group_centers)
    ax.set_xticklabels(algorithm_names, fontsize=10, rotation=20, ha="right", rotation_mode="anchor")

    ax.set_xlabel('FL Algorithm')
    ax.set_ylabel('AUC-PR')

    ax.grid(axis='y')
    ax.set_ylim(0.75, 1)
    # ax.set_ylim(0.7, 0.95)
    ax.legend(loc='lower right', framealpha=1)

    ax.set_title(f'Algorithm Performance as Hetergeniety Increase (N={NC}, {dataset})')

    plt.tight_layout()

    plt.savefig(os.path.join(tables_path, f'hetrogeniety_visualization_among_clients_n={NC}_{dataset}.png'), dpi=300)

import pandas as pd
import matplotlib.pyplot as plt
import os

# where to output figures and tables
BASE_OUTPUT_PATH = '/home/bdanek2/multi-omics-pdd-FL-study/federated_learning_multi_modality_ancestry/multi_modality_fl/results/repro-generated_figures_tables'

dataset = "PDBP"
metric = "auc_precision_recall"

def heterogenity_delta(dataset: str, tables_path: str, metric: str = "auc_precision_recall"):

    us_table = (f'k_averaged_results_table_k_all {dataset} Uniform Stratified test.csv', True)
    ur_table = (f'k_averaged_results_table_k_all {dataset} Uniform Random test.csv', False)
    lr_table = (f'k_averaged_results_table_k_all {dataset} Linear Random test.csv', False)

    tables = []
    for table_file_name, stratified in [us_table, ur_table, lr_table]: # remove LR
        table = pd.read_csv(os.path.join(tables_path, table_file_name))
        table = pd.concat([table[table['num_clients'] == 2], table[table['num_clients'] == 4]])
        # display(table.columns)
        table['algorithm_name'] = table['algorithm_name'].apply(lambda x: x.replace('Classifier', '') if type(x) == str else x)
        table = table.round(4)
        table[f"str_{metric}"] = table[metric].astype(str) + ' ± ' + table[f"{metric}_std"].astype(str)

        table['stratified'] = stratified
        tables.append(table)

    tables = pd.concat(tables)

    # table_figure = tables.set_index(['algorithm_name']).pivot(columns='num_clients')[metric]
    table_figure = tables.set_index(['num_clients', 'algorithm_name']).pivot(columns=['stratified', 'split_method'])[f"str_{metric}"]

    table_figure.to_csv(os.path.join(tables_path, f'hetrogeniety_among_clients_{dataset}.csv'))
    
    print("=====================================")
    print(f'hetrogeniety_among_clients_{dataset}')
    print(table_figure.to_markdown())

    cli_2 = tables.set_index(['num_clients', 'algorithm_name', 'split_method', 'stratified']).loc[2]
    cli_4 = tables.set_index(['num_clients', 'algorithm_name', 'split_method', 'stratified']).loc[4]
    delta_table = pd.concat([cli_4[metric] - cli_2[metric], (cli_4[f'{metric}_std'] - cli_2[f'{metric}_std']).abs()], axis=1).reset_index()
    raw_dt = delta_table.copy()
    # display(delta_table)
    delta_table[metric] = delta_table[metric].round(4).astype(str) + '±' + delta_table[f'{metric}_std'].round(4).astype(str)
    delta_table = delta_table.set_index(['algorithm_name']).pivot(columns=['stratified', 'split_method'])[metric]
    # display(delta_table)

    delta_table.to_csv(os.path.join(tables_path, f'hetrogeniety_among_clients_delta_{dataset}.csv'))

    print("")
    print(f'hetrogeniety_among_clients_delta_{dataset}')
    print(delta_table.to_markdown())

    raw_dt = raw_dt.set_index(['algorithm_name', 'split_method', 'stratified'])

    colors = [
        'gray',
        'salmon',
        'red'
    ]


    # plot the delta table

    bar_width = 0.35
    group_centers = []

    f, ax = plt.subplots()
    f.set_figheight(4)
    f.set_figwidth(10)

    t = tables[tables['num_clients'] == 2]
    std_ = f"{metric}_std"

    loc = 0.5
    alg_name = 'FedAvg LR'
    score = raw_dt.loc[alg_name, 'uniform', True][metric]
    error = raw_dt.loc[alg_name, 'uniform', True][std_]
    ax.bar(loc, score, bar_width, label='Uniform Stratified', color=colors[0], yerr=error, align='edge')

    loc = 1
    score = raw_dt.loc[alg_name, 'uniform', False][metric]
    error = raw_dt.loc[alg_name, 'uniform', False][std_]
    ax.bar(loc, score, bar_width, label='Uniform Random', color=colors[1], yerr=error, align='edge')

    loc = 1.5
    score = raw_dt.loc[alg_name, 'linear', False][metric]
    error = raw_dt.loc[alg_name, 'linear', False][std_]
    ax.bar(loc, score, bar_width, label='Linear Random', color=colors[2], yerr=error, align='edge')
    group_centers.append(1 + bar_width / 2)


    loc = 2.5
    alg_name = 'FedProx μ = 0 LR'
    score = raw_dt.loc[alg_name, 'uniform', True][metric]
    error = raw_dt.loc[alg_name, 'uniform', True][std_]
    ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    loc = 3.0
    score = raw_dt.loc[alg_name, 'uniform', False][metric]
    error = raw_dt.loc[alg_name, 'uniform', False][std_]
    ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    loc = 3.5
    score = raw_dt.loc[alg_name, 'linear', False][metric]
    error = raw_dt.loc[alg_name, 'linear', False][std_]
    ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    group_centers.append(3 + bar_width / 2)


    loc = 4.5
    alg_name = 'FedProx μ = 2 LR'
    score = raw_dt.loc[alg_name, 'uniform', True][metric]
    error = raw_dt.loc[alg_name, 'uniform', True][std_]
    ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    loc = 5.0
    score = raw_dt.loc[alg_name, 'uniform', False][metric]
    error = raw_dt.loc[alg_name, 'uniform', False][std_]
    ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    loc = 5.5
    score = raw_dt.loc[alg_name, 'linear', False][metric]
    error = raw_dt.loc[alg_name, 'linear', False][std_]
    ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    group_centers.append(5 + bar_width / 2)


    # loc = 6.5
    # alg_name = 'FedProx μ = 5 LR'
    # score = raw_dt.loc[alg_name, 'uniform', True][metric]
    # error = raw_dt.loc[alg_name, 'uniform', True][std_]
    # ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    # loc = 7.0
    # score = raw_dt.loc[alg_name, 'uniform', False][metric]
    # error = raw_dt.loc[alg_name, 'uniform', False][std_]
    # ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    # loc = 7.5
    # score = raw_dt.loc[alg_name, 'linear', False][metric]
    # error = raw_dt.loc[alg_name, 'linear', False][std_]
    # ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    # group_centers.append(7 + bar_width / 2)


    # loc = 8.5
    # alg_name = 'FedProx μ = 8 LR'
    # score = raw_dt.loc[alg_name, 'uniform', True][metric]
    # error = raw_dt.loc[alg_name, 'uniform', True][std_]
    # ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    # loc = 9.0
    # score = raw_dt.loc[alg_name, 'uniform', False][metric]
    # error = raw_dt.loc[alg_name, 'uniform', False][std_]
    # ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    # loc = 9.5
    # score = raw_dt.loc[alg_name, 'linear', False][metric]
    # error = raw_dt.loc[alg_name, 'linear', False][std_]
    # ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    # group_centers.append(9 + bar_width / 2)


    loc = 6.5
    alg_name = 'FedAvg MLP'
    score = raw_dt.loc[alg_name, 'uniform', True][metric]
    error = raw_dt.loc[alg_name, 'uniform', True][std_]
    ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    loc = 7
    score = raw_dt.loc[alg_name, 'uniform', False][metric]
    error = raw_dt.loc[alg_name, 'uniform', False][std_]
    ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    loc = 7.5
    score = raw_dt.loc[alg_name, 'linear', False][metric]
    error = raw_dt.loc[alg_name, 'linear', False][std_]
    ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    group_centers.append(7 + bar_width / 2)


    loc = 8.5
    alg_name = 'FedProx μ = 0 MLP'
    score = raw_dt.loc[alg_name, 'uniform', True][metric]
    error = raw_dt.loc[alg_name, 'uniform', True][std_]
    ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    loc = 9
    score = raw_dt.loc[alg_name, 'uniform', False][metric]
    error = raw_dt.loc[alg_name, 'uniform', False][std_]
    ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    loc = 9.5
    score = raw_dt.loc[alg_name, 'linear', False][metric]
    error = raw_dt.loc[alg_name, 'linear', False][std_]
    ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    group_centers.append(9 + bar_width / 2)


    loc = 10.5
    alg_name = 'FedProx μ = 2 MLP'
    score = raw_dt.loc[alg_name, 'uniform', True][metric]
    error = raw_dt.loc[alg_name, 'uniform', True][std_]
    ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    loc = 11
    score = raw_dt.loc[alg_name, 'uniform', False][metric]
    error = raw_dt.loc[alg_name, 'uniform', False][std_]
    ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    loc = 11.5
    score = raw_dt.loc[alg_name, 'linear', False][metric]
    error = raw_dt.loc[alg_name, 'linear', False][std_]
    ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    group_centers.append(11 + bar_width / 2)


    # loc = 16.5
    # alg_name = 'FedProx μ = 5 MLP'
    # score = raw_dt.loc[alg_name, 'uniform', True][metric]
    # error = raw_dt.loc[alg_name, 'uniform', True][std_]
    # ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    # loc = 17
    # score = raw_dt.loc[alg_name, 'uniform', False][metric]
    # error = raw_dt.loc[alg_name, 'uniform', False][std_]
    # ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    # loc = 17.5
    # score = raw_dt.loc[alg_name, 'linear', False][metric]
    # error = raw_dt.loc[alg_name, 'linear', False][std_]
    # ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    # group_centers.append(17 + bar_width / 2)



    # loc = 18.5
    # alg_name = 'FedProx μ = 8 MLP'
    # score = raw_dt.loc[alg_name, 'uniform', True][metric]
    # error = raw_dt.loc[alg_name, 'uniform', True][std_]
    # ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    # loc = 19
    # score = raw_dt.loc[alg_name, 'uniform', False][metric]
    # error = raw_dt.loc[alg_name, 'uniform', False][std_]
    # ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    # loc = 19.5
    # score = raw_dt.loc[alg_name, 'linear', False][metric]
    # error = raw_dt.loc[alg_name, 'linear', False][std_]
    # ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    # group_centers.append(19 + bar_width / 2)



    loc = 12.5
    alg_name = 'FedAvg SGD'
    score = raw_dt.loc[alg_name, 'uniform', True][metric]
    error = raw_dt.loc[alg_name, 'uniform', True][std_]
    ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    loc = 13.0
    score = raw_dt.loc[alg_name, 'uniform', False][metric]
    error = raw_dt.loc[alg_name, 'uniform', False][std_]
    ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    loc = 13.5
    score = raw_dt.loc[alg_name, 'linear', False][metric]
    error = raw_dt.loc[alg_name, 'linear', False][std_]
    ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    group_centers.append(13 + bar_width / 2)


    loc = 14.5
    alg_name = 'FedAvg XGBRF'
    score = raw_dt.loc[alg_name, 'uniform', True][metric]
    error = raw_dt.loc[alg_name, 'uniform', True][std_]
    ax.bar(loc, score, bar_width, color=colors[0], yerr=error, align='edge')

    loc = 15
    score = raw_dt.loc[alg_name, 'uniform', False][metric]
    error = raw_dt.loc[alg_name, 'uniform', False][std_]
    ax.bar(loc, score, bar_width, color=colors[1], yerr=error, align='edge')

    loc = 15.5
    score = raw_dt.loc[alg_name, 'linear', False][metric]
    error = raw_dt.loc[alg_name, 'linear', False][std_]
    ax.bar(loc, score, bar_width, color=colors[2], yerr=error, align='edge')
    group_centers.append(15 + bar_width / 2)


    ax.set_xticks(group_centers)
    ax.set_xticklabels(['FedAvg LR', 'FedProx μ = 0.5 LR', 'FedProx μ = 2 LR', 'FedAvg MLP', 'FedProx μ = 0.5 MLP', 'FedProx μ = 2 MLP', 'FedAvg SGD', 'FedAvg XGBRF'], fontsize=10, rotation=20, ha="right", rotation_mode="anchor")

    ax.set_xlabel('FL Algorithm')
    ax.set_ylabel('AUC-PR')

    ax.grid(axis='y')
    ax.set_ylim(-0.1, 0.1)
    ax.legend()

    ax.set_title(f'Difference in Score Between As Federation Size Increases (N = Δ(2, 4), {dataset})')

    plt.tight_layout()

    plt.savefig(os.path.join(tables_path, f'hetrogeniety_visualization_among_clients_delta_{dataset}.png'), dpi=300)

if __name__ == "__main__":
    """
    COMPUTE ALL THE PAPER FIGURES AND TABLES
    """

    # UPDATE THE BELOW WITH YOUR OWN PATH
    # base of results directory
    RESULTS_DIR = "/home/bdanek2/multi-omics-pdd-FL-study/federated_learning_multi_modality_ancestry/multi_modality_fl/results/experiment_logs"

    # UPDATE THE BELOW WITH YOUR OWN PATH
    # where to output figures and tables
    BASE_OUTPUT_PATH = '/home/bdanek2/multi-omics-pdd-FL-study/federated_learning_multi_modality_ancestry/multi_modality_fl/results/repro-generated_figures_tables'
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

    # once the overall table is generated for a split method, we can generate tables from the paper.
    # table papers are from the uniform stratified split method (as can be seen in the string literals below)
    INTERNAL_RESULTS_TABLE = 'k_averaged_results_table_k_all PPMI Uniform Stratified test.csv'
    EXTERNAL_RESULTS_TABLE = 'k_averaged_results_table_k_all PDBP Uniform Stratified test.csv'

    """
    Each of the below are for a different heterogeniety/FL configuration
    """

    external_results_filename = [f'global_experiment_runner_k{i}_external test' for i in range(0, 6)]
    internal_results_filenames = [f'global_experiment_runner_k{i}_internal test' for i in range(0, 6)]

    """
    Each of the below are for a different heterogeniety/FL configuration
    """
    # fig 3, 4, 5
    results_directory = f'{RESULTS_DIR}/manual_experiments_linear_non_strat'
    generate_plots(BASE_OUTPUT_PATH, "PDBP Linear Random", [0.5, 1], results_directory, external_results_filename, 'csv')
    generate_plots(BASE_OUTPUT_PATH, "PPMI Linear Random", [0.5, 1], results_directory, internal_results_filenames, 'csv')

    # fig 5
    results_directory = f'{RESULTS_DIR}/manual_experiments_uniform_strat'
    generate_plots(BASE_OUTPUT_PATH, "PDBP Uniform Stratified", [0.725, 0.95], results_directory, external_results_filename, 'csv')
    generate_plots(BASE_OUTPUT_PATH, "PPMI Uniform Stratified", [0.65, 1], results_directory, internal_results_filenames, 'csv')

    # fig 5
    results_directory = f'{RESULTS_DIR}/manual_experiments_uniform_non_strat'
    generate_plots(BASE_OUTPUT_PATH, "PDBP Uniform Random", [0.5, 1], results_directory, external_results_filename, 'csv')
    generate_plots(BASE_OUTPUT_PATH, "PPMI Uniform Random", [0.5, 1], results_directory, internal_results_filenames, 'csv')

    # generate table 1
    dataset = 'PPMI'
    processed_data_ppmi = f'{BASE_OUTPUT_PATH}/k_averaged_results_table_k_all {dataset} Uniform Stratified test.csv'
    ppmi_table_output = f'{BASE_OUTPUT_PATH}/{dataset}_n_2_clients_cleaned_combined.csv'
    create_tables(dataset, processed_data_ppmi, ppmi_table_output)

    # generate table 2
    dataset = 'PDBP'
    processed_data_pdbp = f'{BASE_OUTPUT_PATH}/k_averaged_results_table_k_all {dataset} Uniform Stratified test.csv'
    pdbp_table_output = f'{BASE_OUTPUT_PATH}/{dataset}_n_2_clients_cleaned_combined.csv'
    create_tables(dataset, processed_data_pdbp, pdbp_table_output)

    table_1 = f'{BASE_OUTPUT_PATH}/PPMI_n_2_clients_cleaned_combined.csv'
    table_2 = f'{BASE_OUTPUT_PATH}/PDBP_n_2_clients_cleaned_combined.csv'


    table_1 = f'{BASE_OUTPUT_PATH}/PPMI_n_2_clients_cleaned_combined.csv'
    print("PPMI table 1, Performance of central and federated learning algorithms for N=2 clients (uniform stratified)")
    print(pd.read_csv(table_1).to_markdown())

    table_2 = f'{BASE_OUTPUT_PATH}/PDBP_n_2_clients_cleaned_combined.csv'
    print("PDBP table 2, Performance of central and federated learning algorithms for N=2 clients (uniform stratified)")
    print(pd.read_csv(table_2).to_markdown())

    table_3 = f'{BASE_OUTPUT_PATH}/global_experiments_k_all PPMI Uniform Stratified test_client_line_plot_table.csv'
    table_4 = f'{BASE_OUTPUT_PATH}/global_experiments_k_all PDBP Uniform Stratified test_client_line_plot_table.csv'

    print("PPMI table 3, AUC-PR vs. number of clients (uniform stratified)")
    print(pd.read_csv(table_3).to_markdown())

    print("")
    print("PDBP table 4, AUC-PR vs. number of clients (uniform stratified)")
    print(pd.read_csv(table_4).to_markdown())

    # create figure 5 (effect of dataset heterogeniety with N=2 clients). 
    make_heterogeniety_figure('PPMI', BASE_OUTPUT_PATH, num_clients=2, metric='auc_precision_recall')
    make_heterogeniety_figure('PDBP', BASE_OUTPUT_PATH, num_clients=2, metric='auc_precision_recall')

    # # create supplementary figure 1
    # make_heterogeniety_figure('PPMI', BASE_OUTPUT_PATH, num_clients=4, metric='auc_precision_recall')
    # make_heterogeniety_figure('PDBP', BASE_OUTPUT_PATH, num_clients=4, metric='auc_precision_recall')

    # create supplementary figure 2
    heterogenity_delta("PDBP", BASE_OUTPUT_PATH, metric)
    heterogenity_delta("PPMI", BASE_OUTPUT_PATH, metric)