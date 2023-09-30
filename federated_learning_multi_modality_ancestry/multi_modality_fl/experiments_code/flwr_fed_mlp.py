import math
import joblib
from typing import Tuple, Union, List, Dict
import numpy as np
import openml
import flwr as fl
import warnings
import sys
import os

from sklearn import metrics

sys.path.append(os.path.abspath('.'))
from multi_modality_fl.utils.data_management import GlobalExperimentsConfiguration, write_json, read_json

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
MLPParams = List[XY]
XYList = List[XY]

def computeAUCPR(y_true, y_pred):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred[:, 1])
    return metrics.auc(recall, precision)

def run_fed_mlp_experiment(repo_parent, current_experiment: GlobalExperimentsConfiguration, fold_idx: int, num_clients: int, split_method: str, num_rounds: int, stratified: bool, num_local_rounds: int, proximal_mu: float, client_lr: float):

    dummy_x = np.random.rand(100, 673)
    dummy_y = np.random.randint(2, size=100)    
    dummy = MLPClassifier().fit(dummy_x, dummy_y)
    dummy_parameters = {
        # 'coef1': dummy.coefs_[0],
        # 'coef2': dummy.coefs_[1],
        # 'intercept1': dummy.intercepts_[0],
        # 'intercept2': dummy.intercepts_[1],
        # 'n_layers': dummy.n_layers_,
        # 'out_activation': dummy.out_activation_,
        # 'n_outputs_': dummy.n_outputs_,
        'label_binerizer_': dummy._label_binarizer,
    }

    score_save_path = f'{repo_parent}/federated_learning_multi_modality_ancestry/multi_modality_fl/experiments/flwr_fed_mlp.json'
    model_save_path = f'{repo_parent}/federated_learning_multi_modality_ancestry/multi_modality_fl/experiments/mlpmodel.joblib'

    # copied from sklearn.uitls.validaiton.py
    import numbers

    def check_random_state(seed):
        """Turn seed into a np.random.RandomState instance.

        Parameters
        ----------
        seed : None, int or instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

        Returns
        -------
        :class:`numpy:numpy.random.RandomState`
            The random state object based on `seed` parameter.
        """
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, numbers.Integral):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState instance" % seed
        )

    def set_model_params(model, params: MLPParams) -> MLPClassifier:
        """Utility function. Sets the parameters of a sklean LogisticRegression model."""
        model.classes_ = np.array([0, 1])
        model.coefs_ = (params[0], params[1])
        model.intercepts_ = (params[2], params[3])
        model.n_layers_ = 3
        model.out_activation_ = 'logistic'
        model.n_outputs_ = 1
        model._label_binarizer = dummy_parameters['label_binerizer_']
        model.n_iter_ = 0
        model.t_ = 0
        model.loss_curve_ = []
        model.best_loss_ = np.inf
        return model

    def set_initial_params(model):
        """Sets initial parameters as zeros Required since model params are
        uninitialized until model.fit is called.

        But server asks for initial parameters from clients at launch. Refer
        to sklearn.linear_model.LogisticRegression documentation for more
        information.
        """
        model.classes_ = np.array([0, 1])
        model.coefs_ = (np.random.rand(673, 100), np.random.rand(100, 1))
        model.intercepts_ = (np.random.rand(100,), np.random.rand(1,))
        model.n_layers_ = 3
        model.out_activation_ = 'logistic'
        model.n_outputs_ = 1
        model._label_binarizer = dummy_parameters['label_binerizer_']
        model.n_iter_ = 0
        model.t_ = 0
        model.loss_curve_ = []
        model.best_loss_ = np.inf

        # if you do not initialize the MLP weights correctly, you will not learn anything
        model._random_state = check_random_state(model.random_state)

        layer_units = [673] + [100] + [model.n_outputs_]

        model.coefs_ = []
        model.intercepts_ = []

        for i in range(model.n_layers_ - 1):
            # Use the initialization method recommended by
            # Glorot et al.
            coef_init, intercept_init = model._init_coef(
                layer_units[i], layer_units[i + 1], np.dtype('float64')
            )
            model.coefs_.append(coef_init)
            model.intercepts_.append(intercept_init)
    

    # Define Flower client
    class MLPClient(fl.client.NumPyClient):

        def __init__(self, x_train, y_train, x_val, y_val) -> None:
            # Create model
            self.model = MLPClassifier(
                random_state=current_experiment.RANDOM_SEED,
                max_iter=300,  # local epoch
                warm_start=True # prevent refreshing weights when fitting
            )

            set_initial_params(self.model)

            self.x_train, self.y_train = x_train, y_train
            self.x_val, self.y_val = x_val, y_val

        def get_model_parameters(self) -> MLPParams:
            """Returns the paramters of a sklearn LogisticRegression model."""
            
            params = [
                self.model.coefs_[0],
                self.model.coefs_[1],
                self.model.intercepts_[0],
                self.model.intercepts_[1]
            ]
            return params

        def get_parameters(self, config):  # type: ignore
            return self.get_model_parameters()

        def fit(self, parameters, config):  # type: ignore
            set_model_params(self.model, parameters)

            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(self.x_train, self.y_train)
            print(f"Training finished for round {config['server_round']}")
            return self.get_model_parameters(), len(self.x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            set_model_params(self.model, parameters)
            try:
                loss = log_loss(self.y_val, self.model.predict_proba(self.x_val))
            except:
                loss = np.inf
            accuracy = self.model.score(self.x_val, self.y_val)
            return loss, len(self.x_val), {"accuracy": accuracy}

    def start_server():

        def get_server_validation() -> XY:
            """Return validation dataset for server-side evaluation."""
            validation_set = current_experiment.validation_dataset
            xy_server_val = current_experiment.as_features_labels(validation_set, current_experiment.LABEL_COL)
            return (xy_server_val[0], xy_server_val[1])

        def get_partitions(num_partitions: int) -> Tuple[XYList, XYList]:
            client_dataframes = current_experiment.get_client_subsets(
                current_experiment.training_dataset,
                num_partitions,
                method=split_method,
                stratified=stratified,
            )

            partitions = []
            for training_client_df, validation_client_df in zip(client_dataframes[0], client_dataframes[1]):
                
                xy_train = current_experiment.as_features_labels(training_client_df, current_experiment.LABEL_COL)
                xy_val = current_experiment.as_features_labels(validation_client_df, current_experiment.LABEL_COL)
                
                # access like 
                # partitions[client_id] -> (x_train, y_train), (x_val, y_val)
                partitions.append(
                    ((xy_train[0], xy_train[1]), (xy_val[0], xy_val[1]))
                )

            return partitions

        def get_client_fn(dataset_partitions: Tuple[XYList, XYList]):
            """Return a function to construc a client.

            The VirtualClientEngine will exectue this function whenever a client is sampled by
            the strategy to participate.
            """

            def client_fn(cid: str) -> fl.client.Client:
                """Construct a FlowerClient with its own dataset partition."""

                # Extract partition for client with id = cid
                ((x_train, y_train), (x_val, y_val)) = dataset_partitions[int(cid)]

                # Create and return client
                return MLPClient(x_train, y_train, x_val, y_val)

            return client_fn

        def fit_round(server_round: int) -> Dict:
            """Send round number to client."""
            return {"server_round": server_round}
    

        def get_evaluate_fn(model: MLPClassifier):
            """Return an evaluation function for server-side evaluation."""

            # Load test data here to avoid the overhead of doing it in `evaluate` itself
            X_server_val, y_server_val = get_server_validation()

            # The `evaluate` function will be called after every round
            def evaluate(server_round, parameters: fl.common.NDArrays, config):
                # Update model with the latest parameters
                set_model_params(model, parameters)
                try:
                    loss = log_loss(y_server_val, model.predict_proba(X_server_val))
                except:
                    loss = np.inf
                
                # update best model
                y_pred = model.predict_proba(X_server_val)
                aucpr = computeAUCPR(y_server_val, y_pred)

                if not os.path.exists(score_save_path):
                    write_json({'max_score': aucpr, 'max_round': server_round, 'max_model_params': model_save_path}, score_save_path)
                    joblib.dump(model, model_save_path)
                else:
                    score_dict = read_json(score_save_path)
                    if aucpr > score_dict['max_score']:
                        score_dict['max_score'] = aucpr
                        score_dict['max_round'] = server_round
                        write_json(score_dict, score_save_path)
                        joblib.dump(model, model_save_path) # overwrite model
                
                accuracy = model.score(X_server_val, y_server_val)

                return loss, {"accuracy": accuracy, 'aucpr': aucpr, 'loss': loss}

            return evaluate

        model = MLPClassifier(
                random_state=current_experiment.RANDOM_SEED,
                max_iter=num_local_rounds,  # local epoch
                warm_start=True # prevent refreshing weights when fitting
            )
        set_initial_params(model)

        if (proximal_mu != None):
            strategy = fl.server.strategy.FedProx(
                min_available_clients=num_clients,
                evaluate_fn=get_evaluate_fn(model),
                on_fit_config_fn=fit_round,
                proximal_mu=proximal_mu
            )
        else:
            strategy = fl.server.strategy.FedAvg(
                min_available_clients=num_clients,
                evaluate_fn=get_evaluate_fn(model),
                on_fit_config_fn=fit_round
            )

        # With a dictionary, you tell Flower's VirtualClientEngine that each
        # client needs exclusive access to these many resources in order to run
        client_resources = {
            "num_cpus": 1,
            "num_gpus": 0,
        }

        partitions = get_partitions(num_clients)
        # c = get_client_fn(partitions)(1)
        # c.fit(c.get_parameters({}), {'server_round': -1})

        # Start simulation
        fl.simulation.start_simulation(
            client_fn=get_client_fn(partitions),
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            # client_ids=[str(x) for x in range(num_clients)],
            client_resources=client_resources,
            # actor_kwargs={
            #     # "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            #     # does nothing if `num_gpus` in client_resources is 0.0
            # },
            ray_init_args={'num_cpus': num_clients, 'num_gpus': 0}
        )

    def evaluate_best_model():
        score_dict = read_json(score_save_path)
        print(f"best round: {score_dict['max_round']}; eval score: {score_dict['max_score']}")
        model = joblib.load(model_save_path)

        for name, dataset in current_experiment.get_combined_test_dataset():

            # if name == 'external test' and fold_idx != 0:
            #     continue
            
            X, y = current_experiment.as_features_labels(dataset, current_experiment.LABEL_COL)

            # validate model performance
            # y_pred = model.predict(X)
            y_pred = model.predict_proba(X)
            print(y)
            print(y_pred)

            # y_pred = np.rint(y_pred)

            aucpr = computeAUCPR(y, y_pred)
            print("AUCPR", name, aucpr)

            print(sum(y_pred) == len(y_pred))
            print(sum(y) == len(y))

            current_experiment.log_raw_experiment_results(
                fold_idx=fold_idx,
                algorithm_name=f'{f"FedProx μ = {int(proximal_mu)}" if proximal_mu else "FedAvg"} MLPClassifier', 
                num_clients=num_clients, 
                split_method=split_method,
                stratified=stratified,
                val_name=name,
                num_rounds=num_rounds,
                num_local_rounds=num_local_rounds,
                client_lr=client_lr,
                proximal_mu=proximal_mu,
                y_true=y, 
                y_pred=y_pred[:, 1]
            )

    start_server()
    evaluate_best_model()
    os.remove(score_save_path) # init every run