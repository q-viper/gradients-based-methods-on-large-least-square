from gradls.experiment_handler import ExperimentHandler
from gradls.vis.matplotlib_vis import MatplotlibVisualizer, MatplotlibVizConfig
from gradls.vis.plot_type import PlotType
from gradls.losses import LossType

from pathlib import Path
import matplotlib.pyplot as plt


expt_handler = ExperimentHandler(root_dir=Path("expt_res_less_noise"))
expt_handler.load_experiments("Exp_100_1000_5")

viz = MatplotlibVisualizer(
    config=MatplotlibVizConfig(
        figsize=(15, 10), title=f"MSE: LR vs Batch vs Optimizers", use_tex=False
    )
)

viz.clear_plots()
plots = expt_handler.make_lr_plots()
for plot in plots:
    viz.append_plot(plot)

viz.generate_plots()
plt.show()

# viz = MatplotlibVisualizer(config=MatplotlibVizConfig(figsize=(15,10),title=f"Loss vs Batch vs Optimizers",
#                                  use_tex=False))
# viz.clear_plots()
# plots = expt_handler.make_loss_plots()
# for plot in plots:
#     viz.append_plot(plot)
# viz.generate_plots()
# plt.show()

selected_loss = LossType.MSE
viz = MatplotlibVisualizer(
    config=MatplotlibVizConfig(
        figsize=(15, 10),
        title=f"Gradients of Best/Worst Flow Across {selected_loss.name}",
        use_tex=False,
    )
)
viz.clear_plots()
plots, grad_plots = expt_handler.plot_best_and_worst_weight_flow(
    plot_type=PlotType.LINE, selected_loss=selected_loss
)
for plot in grad_plots:
    viz.append_plot(plot)

viz.generate_plots()
plt.show()

viz = MatplotlibVisualizer(
    config=MatplotlibVizConfig(
        figsize=(15, 10),
        title=f"Best/Worst Weight Flow Across {selected_loss.name}",
        use_tex=False,
    )
)
viz.clear_plots()
for plot in plots:
    viz.append_plot(plot)

viz.generate_plots()
plt.show()
