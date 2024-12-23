"""
A module to run an experiment from a saved experiment file.
"""

import argparse
from gradls.experiment import Experiment, ExperimentConfig, Optimizer, Runner
from gradls.vis.matplotlib_vis import MatplotlibVisualizer, MatplotlibVizConfig
import numpy as np
from pathlib import Path
from gradls.datagenerator import DataGenerator, DataGeneratorConfig, MyDataset
from gradls.losses import Loss, LossType

# Parse arguments
parser = argparse.ArgumentParser(description="Read exp.npy and run it.")
parser.add_argument("exp", type=str, help="Path to exp.npy")

# Run experiment
args = parser.parse_args()
exp_path = Path(args.exp)
# Load experiment
exp = np.load(exp_path, allow_pickle=True).item()
# Run experiment
exp.train()
exp.log_expt()
# Save experiment
np.save(exp_path, exp)
