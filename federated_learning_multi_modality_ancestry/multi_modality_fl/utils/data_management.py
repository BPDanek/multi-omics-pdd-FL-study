import os
import sys
from pathlib import Path

REPO_PARENT = Path(__file__).parents[3]

sys.path.append(os.path.abspath("."))
sys.path.append(str(REPO_PARENT))

from functools import lru_cache
import itertools
import json
import random
from typing import Any, Callable, Dict, List, Tuple, Union
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, average_precision_score, fbeta_score, log_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import torch
import collections
import h5py

# utility functions


def build_full_path(base_path, subset=None):
    assert subset != None, "Must provide subset"
    return os.path.join(base_path, subset)


def write_json(data: Any, path: str):
    """Dump json file at path with indent=4"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def read_json(path: str):
    if not os.path.isfile(path):
        raise ValueError(f"{path} does not exist!")
    with open(path, "r") as f:
        return json.load(f)


def get_h5_data_keys(file_path):
    f = h5py.File(file_path, 'r')
    keys = list(f.keys())
    f.close()
    return keys

# used for safe displaying of data in ipynb cells
# used in exploration and baselines file


def drop_id(dataset: pd.DataFrame):
    return dataset.drop(columns=['ID'])


class GlobalExperimentsConfiguration:

    # num_rounds = 20

    RANDOM_SEED = 2

    K = 6

    STANDARD = 'standard'
    SKLEARN = 'sklearn'

    MULTIMODALITY = 'multi_modality'
    MULTIMODALITY_LABEL_COL = 'PHENO'
    MULTIMODALITY_DATASET_FILES = {
        'clinicodemogrpahic_ppmi': 'Clinicodemographic/PPMI_Only_clinical.dataForML.h5',
        'transcriptomics_ppmi': 'TRANSCRIPTOMICS_p1E2/PPMI_Only_transcriptomics_only-p1E2.dataForML.h5',
        'genetics_ppmi': 'GENETICS_p1E5/PPMI_Only_genetics_p1E5_with_PRS-MAF05.dataForML.h5',

        'combined_ppmi': 'Combined_G1E5_O1E2/PPMI-genetic_p1E5_omic_p1E2.dataForML.h5',

        'validation_pdbp': 'Validation/validate-PDBP-genetic_p1E5_omic_p1E2.dataForML.h5'
    }

    MULTIANCESTRY = 'multi_ancestry'
    MULTIANCESTRY_DATASET_FOLDER = ''
    MULTIANCESTRY_DATASET_FILES = {}

    # used for outputting tables
    metadata_column_names = ['algorithm_name',
                             'num_clients', 'split_method', 'val_name']

    def __init__(self, base_path: str, experiment_name: str, random_seed=None):
        if random_seed:
            self.RANDOM_SEED = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.raw_experiment_logs = collections.defaultdict(list)

        # depracate
        self.experiment_results = {}
        self.experiment_name = experiment_name

        # the path which will contain all intermediate experiment files
        self.experiment_path = os.path.join(base_path, experiment_name)
        os.makedirs(self.experiment_path, exist_ok=True)

        self.plots_path = os.path.join(self.experiment_path, 'plots')
        os.makedirs(self.plots_path, exist_ok=True)

        # avoid duplicating external_val dataset
        self.external_val_recorded = False

    # @lru_cache(maxsize=10)
    def _get_raw_dataset(self, path: str, drop: str):
        """load the dataset so you can get its size. @lru_cache decorated for speed."""
        assert '.h5' in path, 'Dataset path does not have a .h5 extension.'
        keys = get_h5_data_keys(path)
        loaded_dataset = pd.read_hdf(path, key=keys[0])
        loaded_dataset = loaded_dataset.drop(columns=['ID'])
        return loaded_dataset

    def _standardize_for_validation(self, dataset_1: pd.DataFrame, dataset_2: pd.DataFrame):
        """use the subset of features (columns) which are present in both datasets"""

        shared_dataset_columns = list(set(dataset_1.columns) & set(
            dataset_2.columns))  # set intersection
        print("shared columns", list(shared_dataset_columns))

        shared_dataset_columns = sorted(shared_dataset_columns)

        not_included = (set(dataset_1.columns) | set(
            dataset_2.columns)) - (set(dataset_1.columns) & set(dataset_2.columns))
        print("non-shared columns", len(not_included), list(not_included))

        print(
            f"shape BEFORE standardization \n{dataset_1.shape} \n {dataset_2.shape}")
        dataset_1 = (dataset_1[shared_dataset_columns])
        dataset_2 = (dataset_2[shared_dataset_columns])
        print(
            f"shape AFTER standardization \n{dataset_1.shape} \n {dataset_2.shape}")

        return dataset_1, dataset_2

    def as_features_labels(self, dataset: pd.DataFrame, label_col: str):
        """make (feature, label) pairs, where `label_col` represents the label col and all others are features.
        Normalize the samples to have balanced value counts if `normalize` is true."""

        features = dataset.drop(columns=[label_col]).copy().to_numpy()
        labels = pd.DataFrame(
            dataset[label_col].copy()).to_numpy().reshape((-1, ))
        return features, labels

    def initialize_data_splits(self, dataset_folder: str, dataset: str, split_method: str = STANDARD):
        assert split_method == self.STANDARD or split_method == self.SKLEARN, f'Unsupported split_method provided. Recieved {split_method}'

        if dataset == self.MULTIMODALITY:

            self.LABEL_COL = self.MULTIMODALITY_LABEL_COL

            # 0. get the dataset sources
            # self.INTERNAL_DATASET = os.path.join(dataset_folder, self.MULTIMODALITY_DATASET_FILES['validation_pdbp'])
            self.INTERNAL_DATASET = os.path.join(
                dataset_folder, self.MULTIMODALITY_DATASET_FILES['combined_ppmi'])

            # a secondary dataset only containing validation data for "extenral validation" (on a dataset from a different distribution as the internal dataset)
            # self.EXTERNAL_DATASET = os.path.join(dataset_folder, self.MULTIMODALITY_DATASET_FILES['combined_ppmi'])
            self.EXTERNAL_DATASET = os.path.join(
                dataset_folder, self.MULTIMODALITY_DATASET_FILES['validation_pdbp'])

            # 1. load the dataset from raw file format
            full_internal_dataset = self._get_raw_dataset(
                self.INTERNAL_DATASET, drop='ID')
            full_external_dataset = self._get_raw_dataset(
                self.EXTERNAL_DATASET, drop='ID')
            print("internal: ", full_internal_dataset.shape)
            print("external: ", full_external_dataset.shape)

            # 2. normalize the feature space the datasets
            # Use the subset of features which are shared between the internal and external dataset
            # ppmi has 675 columns, but the combined pdbp dataset has 715. Drop the 40 extra columns from pdbp
            self.full_internal_dataset, self.full_external_dataset = self._standardize_for_validation(
                full_internal_dataset, full_external_dataset)
            print("full internal", self.full_internal_dataset.info())
            print("full external", self.full_external_dataset.info())
            # 3. compute k folds for the internal dataset

            self._generate_stratified_k_folds(self.full_internal_dataset)
            self.full_external_dataset = self.full_external_dataset.sample(
                frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)

        elif dataset == self.MULTIANCESTRY:
            assert False, "multi ancestry not implemented yet"

        else:
            assert False, f"Unsupported dataset type provided; received {dataset}"

        return self

    def _generate_stratified_k_folds(self, df: pd.DataFrame):
        """Generate k folds of the dataset and store them in the class variable `self.k_fold_indeces`"""
        k_fold_indeces: Dict[int, pd.DataFrame] = dict()

        for _, group in df.groupby('PHENO'):

            fold_len = len(group) // self.K
            start = 0
            for fold in range(0, self.K):
                end = start + fold_len if fold != self.K - 1 else len(group)

                fold_data = group.iloc[start:end]
                if fold not in k_fold_indeces:
                    k_fold_indeces[fold] = fold_data
                else:
                    k_fold_indeces[fold] = pd.concat(
                        [k_fold_indeces[fold], fold_data])

                start = end

        # sanity check, since this is such a crucial part of the experimental design
        for i, subset_i in k_fold_indeces.items():
            for j, subset_j in k_fold_indeces.items():
                if i == j:
                    continue
                assert set(subset_i.index) & set(
                    subset_j.index) == set(), "folds have overlapping indeces"

        # all partitions must have approximately similar startification
        stratifications_across_folds = [fold_values['PHENO'].value_counts(
        )[0] / fold_values['PHENO'].value_counts()[1] for fold_values in k_fold_indeces.values()]
        assert np.std(
            stratifications_across_folds) < 0.03, f"folds do not have balanced startification: {stratifications_across_folds}; std: {np.std(stratifications_across_folds)}"

        self.k_fold_indeces = k_fold_indeces

    def set_fold(self, fold_idx: int):
        """Use the provided `fold_idx` as the holdout dataset, use the rest in training
f
        Args:
            fold_idx: the fold which is the holdout dataset

        Returns:
            holdout_dataset, training_dataset
        """

        holdout_idx = fold_idx
        holdout_dataset = self.k_fold_indeces[holdout_idx]

        trainig_folds = []
        for fold_idx, fold in self.k_fold_indeces.items():
            if fold_idx != holdout_idx:
                trainig_folds.append(fold)

        training_dataset = pd.concat(trainig_folds)

        self.training_dataset = training_dataset.sample(frac=1, replace=False, random_state=self.RANDOM_SEED).reset_index(drop=True)
        self.internal_test_dataset = holdout_dataset.sample(frac=1, replace=False, random_state=self.RANDOM_SEED).reset_index(drop=True)

        # no data loss
        assert set(self.training_dataset.itertuples(index=False, name=None)) | set(self.internal_test_dataset.itertuples(index=False, name=None)) == set(self.full_internal_dataset.itertuples(index=False, name=None))
        
        # the training dataset and holdout are disjoint
        assert set(self.training_dataset.itertuples(index=False, name=None)) & set(self.internal_test_dataset.itertuples(index=False, name=None)) == set()

        self.set_validation_dataset() # create the self.validation_dataset var

        # these are checked in the set_validation_dataset method, but we check them here for possible debugging help
        # no data loss
        assert set(self.training_dataset.itertuples(index=False, name=None)) | set(self.validation_dataset.itertuples(index=False, name=None)) == set(training_dataset.itertuples(index=False, name=None))
        # the training dataset and validation dataset are disjoint
        assert set(self.training_dataset.itertuples(index=False, name=None)) & set(self.validation_dataset.itertuples(index=False, name=None)) == set()

    def set_validation_dataset(self, ratios=[0.8, 0.2]):
        """Splits the current training dataset by the ratios, setting `self.training_dataset` to the first split, and `self.validation_dataset` to the second"""
        new_datasets = self._do_stratified_trainval_split(self.training_dataset, ratios=ratios)
        
        assert len(
            new_datasets) == 2, f"Validaiton splitting failed; expected 2 new datasets, got {len(new_datasets)}"

        self.training_dataset, self.validation_dataset = new_datasets[0], new_datasets[1]

    # def _generate_k_fold_indeces(self, dataset: pd.DataFrame, k: int):
    #     """generate the indeces for the train/test split and store them in class instance variables"""
    #     kf = KFold(n_splits=k, shuffle=True, random_state=self.RANDOM_SEED)
    #     self.train_fold_indices = []
    #     self.test_fold_indeces = []
    #     self.val_fold_indices = []
    #     for train_index, val_index in kf.split(dataset):
    #         # we want test set to be 10% of overall dataset
    #         # 0.8 * x = 0.1 * 1
    #         # x = 0.1/0.8
    #         train_test = train_test_split(train_index, test_size=0.125, random_state=self.RANDOM_SEED)
    #         train_index, test_index = train_test[0], train_test[1]
    #         self.train_fold_indices.append(train_index)
    #         self.test_fold_indeces.append(test_index)
    #         self.val_fold_indices.append(val_index)

    #         print("set indeces", set(train_index) & set(val_index))

    # def set_train_dataset(self, fold_idx: int):
    #     """set the class instance variabel `training_dataset` to the training subset for the provided fold"""
    #     self.training_dataset = self.full_internal_dataset.iloc[self.train_fold_indices[fold_idx]].reset_index(drop=True)
    #     self.test_dataset = self.full_internal_dataset.iloc[self.test_fold_indeces[fold_idx]].reset_index(drop=True)

    def get_combined_test_dataset(self):
        return [
            ("internal test", self.internal_test_dataset.reset_index(drop=True)),
            ("external test", self.full_external_dataset.reset_index(drop=True))
        ]

    # depracated
    # def _split_dataframe(self, dataset: pd.DataFrame, ratios: List[int], shuffle: bool, as_intervals: bool) ->  List[Union[pd.DataFrame, Tuple[int]]]:
    #     """
    #     Split the internal dataset by ratios & handle shuffling.
    #     Returns either indeces of the dataset splits, or the dataset subsets depending on parameter `as_intervals`.
    #     If `as_intervals` is set, one cannot shuffle the dataset, because it would be redundant and is probably a mistake on the programmers part.
    #     """

    #     if shuffle:
    #         assert shuffle, "shuffle depracted"
    #         # assert as_intervals == False, 'There is no need to shuffle the df if we are just returning indeces.'
    #         # dataset = dataset.sample(frac=1).reset_index(drop=True)

    #     indeces = [0]
    #     for i, ratio in enumerate(ratios):

    #         last_split = (i == len(ratios) - 1)
    #         next_index = indeces[-1] + int(ratio * len(dataset)) if not last_split else len(dataset)
    #         indeces.append(next_index)

    #     # make sure we don't incorrectly calculate the splits for some reason
    #     assert sum([indeces[i+1] - indeces[i] for i in range(0, len(indeces) - 1)]) == len(dataset), f"Dataset splits do not correctly split the dataset. Expected {len(dataset)} received {sum([indeces[i+1] - indeces[i] for i in range(0, len(indeces) - 1)])}; received ratios {ratios}"

    #     result = []
    #     for i in range(0, len(indeces) - 1):
    #         start, end = indeces[i], indeces[i + 1]

    #         if as_intervals:
    #             result.append((start, end))
    #         else:
    #             result.append(dataset[start: end])

    #     return result

    def get_client_subsets(self, dataset: pd.DataFrame, num_clients: int, method: str, stratified: bool) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        client_subsets = None
        if stratified:
            client_subsets = self.get_stratified_client_subsets(dataset, num_clients, method)
        else:
            client_subsets = self.get_uniform_random_client_subsets(dataset, num_clients, method)
        
        for i, trainval_split in enumerate(client_subsets):

            for a, b in itertools.combinations(trainval_split, 2):
                assert set(a.itertuples(index=False, name=None)) & set(b.itertuples(index=False, name=None)) == set(), f"trainval splits have overlapping indeces for {'val' if i else 'train'}"    

        return client_subsets
    
    def get_uniform_random_client_subsets(self, dataset: pd.DataFrame, num_clients: int, method: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        
        ratios = self.method_to_ratios(method=method, num_clients=num_clients)

        shuffled_dataset = dataset.sample(frac=1, replace=False, random_state=self.RANDOM_SEED)
        indeces = []
        for i, ratio in enumerate(ratios):
            
            # we do not need to include the last ratio, since it is implicit numerically as (1 - sum(ratios)
            if i == len(ratios) - 1:
                break

            start = 0 if len(indeces) == 0 else indeces[-1]
            next_index = start + int(ratio * len(shuffled_dataset))
            indeces.append(next_index)

        print(indeces)

        client_splits = np.split(shuffled_dataset, indeces)
        
        validation_subsets = []
        training_subsets = []
        for samples in client_splits:
            size_of_validation = int(len(samples) * 0.2)
            indeces = list(range(0, len(samples)))
            np.random.shuffle(indeces)
            
            validation_subsets.append(samples.iloc[indeces[:size_of_validation]])
            training_subsets.append(samples.iloc[indeces[size_of_validation:]])

        return (training_subsets, validation_subsets)

    def get_stratified_client_subsets(self, dataset: pd.DataFrame, num_clients: int, method: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        ratios = self.method_to_ratios(method=method, num_clients=num_clients)
        check_proportions = method == 'uniform'
        return self.stratified_split(df=dataset, column=self.MULTIMODALITY_LABEL_COL, ratios=ratios, check_proportions=check_proportions)

    def _do_stratified_trainval_split(self, dataset: pd.DataFrame, ratios: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        assert len(ratios) == 2, f"Expected 2 ratios, received {len(ratios)}"
        splits = {}
        groups = dataset.groupby(self.LABEL_COL)
        for _, group in groups:
            shuffled_group = group.sample(
                frac=1, replace=False, random_state=self.RANDOM_SEED)

            indeces = []
            for i, ratio in enumerate(ratios):
                
                # we do not need to include the last ratio, since it is implicit numerically as (1 - sum(ratios)
                if i == len(ratios) - 1:
                    break

                start = 0 if len(indeces) == 0 else indeces[-1]
                next_index = start + int(ratio * len(shuffled_group))
                indeces.append(next_index)

            print(_, indeces)

            # split by the ratios provided as a parameter
            split_samples = np.split(shuffled_group, indeces)
            assert len(split_samples) == len(ratios), f"Expected {len(ratios)} splits, received {len(split_samples)}"

            # put the stratified split into a dictionary corresponding to its group
            for split_index, split in enumerate(split_samples):
                if split_index not in splits:
                    splits[split_index] = split
                else:
                    splits[split_index] = pd.concat([splits[split_index], split])

        # shuffle the splits after stratifying, otherwise we will have a block of each label
        for i, split in splits.items():
            splits[i] = split.sample(frac=1, replace=False, random_state=self.RANDOM_SEED).reset_index(drop=True)

        assert len(splits) == len(ratios), f"Expected {len(ratios)} splits, received {len(splits)}"
        
        assert len(splits[0]) + len(splits[1]) == len(dataset), f"Expected {len(dataset)} samples, received {len(splits[0]) + len(splits[1])}"
        assert set(splits[0].itertuples(index=False, name=None)) & set(splits[1].itertuples(index=False, name=None)) == set(), "validation and training subsets have overlapping indeces"
        
        assert len(splits[0]) - len(dataset.sample(frac=ratios[0], replace=False, random_state=self.RANDOM_SEED)) < 0.05, f"Expected {len(dataset) * ratios[0]} samples in split 0, received {len(splits[0])}"

        vc0 = splits[0]['PHENO'].value_counts()
        vc1 = splits[1]['PHENO'].value_counts()
        assert vc0[0] / vc0[1] - vc1[0] / vc1[1] < 0.05, f"Value counts of stratified dataset inconsistnet. {vc0[0]}:{vc0[1]} vs {vc1[0]}:{vc1[1]}"
        
        return (splits[0], splits[1])
        

    def stratified_split(self, df: pd.DataFrame, column: str, ratios: float, check_proportions=False) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        samples = []
        groups = df.groupby(column)
        for _, group in groups:
            shuffled_group = group.sample(
                frac=1, replace=False, random_state=self.RANDOM_SEED)

            indeces = []
            for i, ratio in enumerate(ratios):
                
                # we do not need to include the last ratio, since it is implicit numerically as (1 - sum(ratios)
                if i == len(ratios) - 1:
                    break

                start = 0 if len(indeces) == 0 else indeces[-1]
                next_index = start + int(ratio * len(shuffled_group))
                indeces.append(next_index)

            print(_, indeces)

            # split by the ratios provided as a parameter
            split_samples = np.split(shuffled_group, indeces)
            assert len(split_samples) == len(ratios), f"Expected {len(ratios)} splits, received {len(split_samples)}"

            samples.append(split_samples)

        # `samples` shape: classes x len(ratios) x size of subset
        # get the subset used for client side validation
        client_validation_samples = {}
        client_training_samples = {}
        for client_splits_by_group in samples: # iterate classes

            for client_id, client_samples_group_label_subset in enumerate(client_splits_by_group): # iterate the `num_clients` subsets within each label class
                size_of_validation = int(len(client_samples_group_label_subset) * 0.2) # 20% of the data at clients used for validation
                indeces = list(range(0, len(client_samples_group_label_subset)))
                np.random.shuffle(indeces)

                validation_subset = client_samples_group_label_subset.iloc[indeces[:size_of_validation]]
                training_subset = client_samples_group_label_subset.iloc[indeces[size_of_validation:]]

                # check for data leaks
                assert set(validation_subset.itertuples(index=False, name=None)) & set(training_subset.itertuples(index=False, name=None)) == set(), "validation and training subsets have overlapping indeces"
                
                if client_id in client_validation_samples and client_id in client_training_samples:
                    client_validation_samples[client_id] = pd.concat([client_validation_samples[client_id], validation_subset])
                    client_training_samples[client_id] = pd.concat([client_training_samples[client_id], training_subset]).reset_index(drop=True)

                    # after stratifying, shuffle the samples, otherwise you have labels = 0 followed by labels = 1
                    client_validation_samples[client_id] = client_validation_samples[client_id].sample(frac=1, replace=False, random_state=self.RANDOM_SEED)
                    client_training_samples[client_id] = client_training_samples[client_id].sample(frac=1, replace=False, random_state=self.RANDOM_SEED).reset_index(drop=True)
                else:
                    client_validation_samples[client_id] = validation_subset
                    client_training_samples[client_id] = training_subset
        
        if check_proportions:
            value_proportions = [subset['PHENO'].value_counts()[0] / subset['PHENO'].value_counts()[1] for subset in client_training_samples.values()]
            print(value_proportions)

            assert np.std(
                value_proportions) < 0.05, f"Value counts of stratified dataset inconsistnet. {value_proportions} : {np.std(value_proportions)}"

        return (list(client_training_samples.values()), list(client_validation_samples.values()))

    def method_to_ratios(self, method: str, num_clients: int):
        assert method in ['uniform', 'linear', 'polynomial',
                          'exponential'], f'Unsupported method specified for client splits. Recieved {method}'

        if method == 'uniform':  # 1
            ratio_vec = np.ones(num_clients)
        elif method == 'linear':  # n
            ratio_vec = np.linspace(1, num_clients, num=num_clients)
        elif method == 'polynomial':  # n^2
            ratio_vec = np.square(np.linspace(1, num_clients, num=num_clients))
        elif method == 'exponential':  # e^n
            ratio_vec = np.exp(np.linspace(1, num_clients, num=num_clients))

        total = sum(ratio_vec)
        ratios = ratio_vec / total
        return ratios

    # def get_client_splits(self, dataset: pd.DataFrame, num_clients: int, method: str, as_intervals=True):
    #     """returns the indeces of the splits on the dataframe. Does not shuffle the dataframe."""
    #     ratios = self.method_to_ratios(method=method, num_clients=num_clients)
    #     intervals = self._split_dataframe(dataset, ratios=ratios, shuffle=False, as_intervals=as_intervals)
    #     return intervals

    def nvflare_multi_site_split_json(
        self,
        data_source_path: List[str],
        validation_data_source_path: List[str],
        site_naming_fn: Callable[..., str],
        site_config_naming_fn: Callable[..., str],
    ) -> List[Tuple[str, dict]]:
        """build the json for client splits for a single nvflare simulation job provided splits"""

        result_files, result_json = [], []
        for index in range(len(data_source_path)):

            json_data = {
                "data_path": data_source_path[index],
                "valid_path": validation_data_source_path[index]
            }

            site_file_name = site_config_naming_fn(index)

            result_files.append(site_file_name)
            result_json.append(json_data)

        print("resulting files configured", result_files)
        return result_files, result_json

    def compute_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
            'roc_auc_score': roc_auc_score(y_true=y_true, y_score=y_pred),
            'average_precision_score': average_precision_score(y_true=y_true, y_score=y_pred),
            'f0.5': fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.5),
            'f1': fbeta_score(y_true=y_true, y_pred=y_pred, beta=1),
            'f2': fbeta_score(y_true=y_true, y_pred=y_pred, beta=2),
            'log_loss': log_loss(y_true=y_true, y_pred=y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true=y_true, y_pred=y_pred)
            # 'num_samples': len(y_true)
        }

    def add_val_result(self, fold_idx: int, algorithm_name: str, num_clients: str, split_method: str, name: str, y_true, y_pred):

        key = (algorithm_name, fold_idx, split_method, num_clients, name)
        assert key not in self.experiment_results, f'This result has already been logged. Current results are {list(self.experiment_results.keys())}, received {key}'

        self.experiment_results[key] = {
            'validation_dataset_name': name,
            'metrics': self.compute_metrics(y_true, y_pred),
            'size': len(y_true),
        }

    def k_fold_results_to_stats(self):
        k_avgs = collections.defaultdict(list)
        for key, val in self.experiment_results.items():
            algorithm_name, _, method, num_clients, name = key
            newKey = (f"{algorithm_name}-{method}-{num_clients}-{name}", name)
            k_avgs[newKey].append(val['metrics']['roc_auc_score'])

        results = collections.defaultdict(dict)
        for key, val in k_avgs.items():
            json_id, dataset = key
            results[dataset][json_id] = {
                'mean': np.mean(val),
                'std': np.std(val)
            }

        return results

    def add_to_kfold_table(self, fold_number: int, algorithm_name: str, num_clients: str, split_method: str, val_name: str, y_true, y_pred):
        assert split_method != 'internal_validation' and split_method != 'external_validation', f'incorrect val name, received {val_name}'
        data = self.compute_metrics(y_true, y_pred)
        row_data = [algorithm_name, num_clients, split_method, val_name]
        row_data.extend(data.values())

        # record validation only once
        if fold_number != 0 and ('external' in val_name):
            return

        if not hasattr(self, 'kfold_table'):
            col_names = self.metadata_column_names.copy()

            all_cols = col_names.copy() + list(data.keys())
            self.kfold_table = pd.DataFrame([row_data], columns=all_cols)
        else:
            self.kfold_table.loc[len(self.kfold_table.index)] = row_data

    def write_results(self, path: str):
        os.makedirs(path, exist_ok=True)
        write_path = os.path.join(path, f"{self.experiment_name}.csv")
        self.kfold_table.to_csv(write_path, index=False)
        return write_path

    def log_raw_experiment_results(self, fold_idx: int, algorithm_name: str, num_clients: str, split_method: str, stratified: bool, val_name: str, num_rounds: int, num_local_rounds: int, client_lr: float, proximal_mu: float, y_true, y_pred):

        key = (fold_idx, algorithm_name, num_clients, split_method, val_name)
        assert key not in self.raw_experiment_logs, f'This result has already been logged. Current results are {list(self.experiment_results.keys())}, received {key}'

        table = []        
        for yt, yp in zip(y_true, y_pred):
            row = {
                'fold_idx': fold_idx,
                'algorithm_name': algorithm_name,
                'num_clients': num_clients,
                'split_method': split_method,
                'stratified': stratified,
                'val_name': val_name,
                'num_samples': len(y_true),
                'y_true': yt,
                'y_pred': yp,
                'num_rounds': num_rounds,
                'num_local_rounds': num_local_rounds,
                'client_lr': client_lr,
                'proximal_mu': proximal_mu
            }
            table.append(row)

        self.raw_experiment_logs[val_name].extend(table)

    def write_raw_experiment_results_as_df(self, fold_idx, path: str):
        os.makedirs(path, exist_ok=True)

        write_paths = []
        for raw_experiment_type in self.raw_experiment_logs.keys():
            write_path = os.path.join(path, f"{self.experiment_name}_k{fold_idx}_{raw_experiment_type}.csv")
            
            if os.path.exists(write_path):
                prev = pd.read_csv(write_path)
                current = pd.DataFrame.from_records(self.raw_experiment_logs[raw_experiment_type])
                prev = pd.concat([prev, current]).reset_index(drop=True)
                prev.to_csv(write_path, index=False)
                print("appended to existing file (logging)")                
            
            else:
                pd.DataFrame.from_records(self.raw_experiment_logs[raw_experiment_type]).to_csv(write_path, index=False)
                print("wrote new file (logging)")
            write_paths.append(write_path)
        
        return write_paths
    
    def computeAUCPR(self, y_true, y_pred):
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred[:, 1])
        return metrics.auc(recall, precision)