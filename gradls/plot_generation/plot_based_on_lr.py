from gradls.experiment_handler import ExperimentHandler
from gradls.vis.matplotlib_vis import (
    MatplotlibVisualizer,
    MatplotlibVizConfig,
    PlotType,
)
from gradls.experiment import Optimizer
from gradls.losses import LossType

from pathlib import Path
import matplotlib.pyplot as plt


expt_handler = ExperimentHandler(root_dir=Path("expt_res_less_noise"))
expt_handler.load_experiments("Exp_100_1000_5")


viz = MatplotlibVisualizer(
    config=MatplotlibVizConfig(
        figsize=(15, 7),
        title="VAL MSE Across Const. LR Optimizers",
        use_tex=False,
        sharey=False,
    )
)
viz.clear_plots()
plots = expt_handler.make_lr_plots(
    loss=LossType.MSE,
    learning_rates=[0.001, 0.01],
    optimizers=[Optimizer.SGD, Optimizer.MOMENTUM, Optimizer.NESTEROV],
)

for i, plot in enumerate(plots):
    viz.append_plot(plot)
    # if i>3:
    #     break

    plot.fixed_ymax = True
    plot.ymax = 4
    if plot.plot_row_col[0] == 0:
        plot.ymax = 13
    elif plot.plot_row_col[0] == 1:
        plot.ymax = 8
    elif plot.plot_row_col[0] == 2:
        plot.ymax = 8
    plot.ymin = 0
    # plot.ymax=None

    plot.linewidth = 1.5
    if "nesterov" in plot.legend:
        plot.linestyle = "--"
    # if "sgd" in plot.legend:
    #     plot.linestyle='-.'
    plot.legend_size = 10
viz.generate_plots("less_noise_const_lr_optimizers.pdf", "pdf")


viz = MatplotlibVisualizer(
    config=MatplotlibVizConfig(
        figsize=(15, 7),
        title="VAL MSE Across Adaptive LR Optimizers",
        use_tex=False,
        sharey=False,
    )
)
viz.clear_plots()
plots = expt_handler.make_lr_plots(
    loss=LossType.MSE,
    optimizers=[
        Optimizer.ADAGRAD,
        Optimizer.RMSPROP,
        Optimizer.ADADELTA,
        Optimizer.ADAM,
    ],
)
for plot in plots:
    viz.append_plot(plot)

    if plot.plot_row_col[0] == 0:
        plot.ymax = 15
    elif plot.plot_row_col[0] == 1:
        plot.ymax = 10
    elif plot.plot_row_col[0] == 2:
        plot.ymax = 8
    plot.ymin = 0
    # plot.ymax=None
    plot.linewidth = 1.5
    plot.fixed_ymax = True
    plot.legend_size = 10


viz.generate_plots("less_noise_var_lr_optimizers.pdf", "pdf")
