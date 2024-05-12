import streamsync as ss
from gradls.losses import LossType, Loss
from gradls.experiment import Experiment, DataGenerator, ExperimentConfig

from dataclasses import dataclass


def create_experiment(state):
    experiment = Experiment(experiment_config=ExperimentConfig())

    return experiment


# Shows in the log when the app starts
print("Hello world!")


initial_state = ss.init_state(
    {
        "my_app": {"title": "GradLS", "version": "0.1.0"},
    }
)
