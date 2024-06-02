import argparse
from gradls.experiment import Experiment, ExperimentConfig, Optimizer, Runner
from gradls.vis.matplotlib_vis import MatplotlibVisualizer, MatplotlibVizConfig
import numpy as np
from pathlib import Path
from gradls.datagenerator import DataGenerator, DataGeneratorConfig, MyDataset
from gradls.losses import Loss, LossType

parser = argparse.ArgumentParser(description="Read exp.npy and run it.")
parser.add_argument("exp", type=str, help="Path to exp.npy")


args = parser.parse_args()
exp_path = Path(args.exp)
exp = np.load(exp_path, allow_pickle=True).item()
exp.train()
exp.log_expt()
np.save(exp_path, exp)
# exp_config = np.load(exp_path.parent / "expt_config.npy", allow_pickle=True).item()
# data_path = exp_path.parent / "data_config.npy"
# viz_path = exp_path.parent / "viz_config.npy"
# exp = Experiment(exp_config)
# data_config = np.load(data_path, allow_pickle=True).item()
# viz_config = np.load(viz_path, allow_pickle=True).item()
# data_gen = DataGenerator(data_config)
# exp = Experiment(config=exp_config)
# exp.load_data(data_gen)
# exp.train()
# logs = {
#     key: runner.logs
#     for key, runner in zip(["train", "val"], [exp.train_runner, exp.val_runner])
# }
# expt_dir = (
#     exp.log_expt()
# )  # Ensure this method is defined and implemented correctly in the Experiment class
# exp.logs = logs
