# look for a json file starting with 'experiment_' in the 'experiments' directory
# then make an experiment config out of it
# then make an experiment out of the config
# then run it

from gradls.losses import LossType, Loss
from gradls.experiment import Experiment, DataGenerator, ExperimentConfig
from gradls.vis.matplotlib_vis import MatplotlibVis, MatplotlibVizConfig
from gradls.webapp.config import webapp_config

import json
import time
from pathlib import Path


def look_experiment_files(path="experiments"):
    experiment_files = []
    for file in Path(path).iterdir():
        if file.is_file() and file.name.startswith("experiment_"):
            experiment_files.append(file)
    return experiment_files


def experiment_handler(look_every=10):
    look_at = webapp_config.experiment_dir
    experiment = None
    while True:
        experiments = look_experiment_files(look_at)
        if len(experiments) > 0:
            for experiment_file in experiments:
                with open(experiment_file) as f:
                    config = json.load(f)
                # in_features = 20
                # out_features = 1
                # viz_config = MatplotlibVizConfig(figsize=(15,10),title="My Exp",
                #                                  use_tex=False)
                # exp_config = ExperimentConfig(name="My Exp", loss=LossType.MSE, viz_config=viz_config,
                #                               num_epochs=100, batch_size=1,learning_rate=0.01, optimizer=Optimizer.ADAM,
                #                               model=None, metrics=[LossType.MAE, LossType.HINGE, LossType.RMSE],
                #                               log_every=1, log_anim=True, anim_fps=10)

                # data = DataGenerator(num_rows=1000, num_cols=in_features, weights=None, biases=None, max_val=100,
                #                      normalize=True, seed=exp_config.seed)

                # exp = Experiment(config=exp_config)
                # exp.load_data(data)
                # exp.train()
                # exp.log_expt()
                experiment_config = config["experiment_config"]
                viz_config = MatplotlibVizConfig(**config["viz_config"])
                experiment_config["viz_config"] = viz_config
                data_config = config["data_config"]
                experiment = Experiment(
                    experiment_config=ExperimentConfig(**experiment_config)
                )
                experiment.load_data(DataGenerator(data_config))
                print("Experiment finished")
                experiment = None
                experiment_file.unlink()

        time.sleep(look_every)
