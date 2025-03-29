import sys
import optuna
import neptune.integrations.optuna as optuna_utils
from perovskiteml.data import ExpandedDataset
from perovskiteml.preprocessing.preprocessor import Preprocessor
from perovskiteml.preprocessing import PrunerFactory
from perovskiteml.utils.config_parser import load_config
from perovskiteml.utils.parameter_sweep import ParameterSweepHandler
from perovskiteml.logging import neptune_context
from perovskiteml.models import ModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


def run(config_path):
    config_dict = load_config(config_path)
    study = optuna.create_study(
        direction=config_dict["hyperparameters"]["direction"],
        study_name=config_dict["hyperparameters"]["study_name"]
    )
    config_sweep = ParameterSweepHandler(config_dict)
    dataset = ExpandedDataset.cache_or_compute(config_dict["data"])

    def objective(trial, config_sweep=config_sweep, dataset=dataset):
        config = config_sweep.suggested_config(trial)
        dataset.reset_features()
        
        pruner = PrunerFactory.create(config["pruning"])
        preprocessor = Preprocessor(config["process"])
        model = ModelFactory.create(config["model"])
        
        pruner.prune(dataset)
        X, y = dataset.split_target()
        X_transformed = preprocessor.preprocess(X, y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_transformed, y, test_size=0.30, random_state=config["experiment"]["seed"]
        )
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_train)
        return root_mean_squared_error(y_pred, y_train)

    with neptune_context(config_dict["logging"]) as neptune_run:
        if neptune_run:
            neptune_callback = optuna_utils.NeptuneCallback(neptune_run)
        else:
            neptune_callback = None
        study.optimize(
            objective,
            callbacks=[neptune_callback] if neptune_callback else None,
            n_trials=config_dict["hyperparameters"]["n_trials"]
        )


if __name__ == "__main__":
    config_path = sys.argv[1]
    run(config_path)
