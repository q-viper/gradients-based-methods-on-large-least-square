from typing import List, Optional, Tuple, Union, field
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from gradls.datagenerator import DataGenerator, DataGeneratorConfig
from gradls.losses import LossType
from gradls.experiment import Experiment, ExperimentConfig, Optimizer
from gradls.vis.matplotlib_vis import MatplotlibVizConfig, Plot, PlotOn, PlotType
from gradls.vis.colors import visible_colors


@dataclass
class ExperimentHandlerConfig:
    root_dir: Path
    data_gen: List[DataGenerator] = field(
        default_factory=lambda: [
            DataGenerator(
                num_rows=1000,
                num_cols=20,
                weights=None,
                biases=None,
                max_val=100,
                normalize=True,
                seed=100,
            )
        ]
    )
    losses: List[LossType] = field(default_factory=lambda: [LossType.MAE])
    metrics: List[LossType] = field(default_factory=lambda: [LossType.MSE])
    num_epochs: List[int] = field(default_factory=lambda: [10])
    batch_sizes: List[int] = field(default_factory=lambda: [64])
    optimizers: List[Optimizer] = field(default_factory=lambda: [Optimizer.ADAM])
    seeds: List[int] = field(default_factory=lambda: [100])
    learning_rates: List[float] = field(default_factory=lambda: [0.1])
    momentums: List[float] = field(default_factory=lambda: [0.0])
    plot_format: str = "png"
    log_anim: bool = True
    log_plots: bool = True
    log_real_params: bool = False
    log_real_data: bool = False
    train_test_split: List[float] = field(default_factory=lambda: [0.1])
    device: str = "cuda"


class ExperimentHandler:
    def __init__(
        self,
        root_dir: Path,
        config: Optional[ExperimentHandlerConfig] = None,
        expt_names: Optional[List[str]] = None,
    ):
        self.config = config
        self.root_dir = root_dir if self.config is None else self.config.root_dir
        self.experiments = {}
        self.losses = {}
        self.metrics = {}
        self.real_weights = {}
        self.real_biases = {}
        self.expt_names = expt_names

        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def make_experiments(self):
        if self.config is None:
            raise ValueError("Please provide a valid config.")

        # count possible combinations
        num_expts = (
            len(self.config.seeds)
            * len(self.config.data_gen)
            * len(self.config.losses)
            * len(self.config.num_epochs)
            * len(self.config.batch_sizes)
            * len(self.config.optimizers)
            * len(self.config.learning_rates)
            * len(self.config.momentums)
        )
        print(
            f"Possible combinations: {len(self.config.seeds)} seeds, {len(self.config.data_gen)} data generators, {len(self.config.losses)} losses, {len(self.config.num_epochs)} epochs, {len(self.config.batch_sizes)} batch sizes, {len(self.config.optimizers)} optimizers, {len(self.config.learning_rates)} learning rates, {len(self.config.momentums)} momentums."
        )

        for seed in self.config.seeds:
            for data_gen in self.config.data_gen:
                expt_name = (
                    f"Exp_{seed}_{data_gen.config.num_rows}_{data_gen.config.num_cols}"
                )
                expt_path = self.root_dir / expt_name
                expt_path.mkdir(parents=True, exist_ok=True)
                i = 0
                for loss in self.config.losses:
                    for num_epochs in self.config.num_epochs:
                        for batch_size in self.config.batch_sizes:
                            for optimizer in self.config.optimizers:
                                for learning_rate in self.config.learning_rates:
                                    for mi, momentum in enumerate(
                                        self.config.momentums
                                    ):
                                        if mi > 0 and optimizer not in [
                                            Optimizer.MOMENTUM,
                                            Optimizer.NESTEROV,
                                        ]:
                                            continue

                                        expt_name = f"Exp_{loss.value}_{num_epochs}_{batch_size}_{optimizer.value}_{learning_rate}"
                                        expt_name = (
                                            expt_name + f"_{momentum}"
                                            if optimizer
                                            in [Optimizer.MOMENTUM, Optimizer.NESTEROV]
                                            else expt_name
                                        )
                                        print(f"Running experiment ({expt_name}).")
                                        exp_config = ExperimentConfig(
                                            name=expt_name,
                                            viz_config=MatplotlibVizConfig(
                                                figsize=(15, 10),
                                                title=f"{expt_name}",
                                                use_tex=False,
                                            ),
                                            num_epochs=num_epochs,
                                            batch_size=batch_size,
                                            loss=loss,
                                            metrics=[
                                                metric
                                                for metric in self.config.metrics
                                                if metric != loss
                                            ],
                                            optimizer=optimizer,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            train_valid_split=0.1,
                                            seed=seed,
                                            log_every=1,
                                            log_dir=expt_path,
                                            log_real_data=self.config.log_real_data,
                                            log_anim=self.config.log_anim,
                                            log_plots=self.config.log_plots,
                                            plot_format=self.config.plot_format,
                                            log_real_params=self.config.log_real_params,
                                        )
                                        exp = Experiment(config=exp_config)
                                        exp.load_data(data_gen)
                                        exp.train()
                                        logs = {
                                            key: runner.logs
                                            for key, runner in zip(
                                                ["train", "val"],
                                                [exp.train_runner, exp.val_runner],
                                            )
                                        }
                                        expt_dir = exp.log_expt()
                                        exp.logs = logs
                                        self.experiments[exp_config.name] = exp
                                        # self.real_weights[exp_config.name] = exp.real_weights
                                        # self.real_biases[exp_config.name] = exp.real_biases
                                        i += 1

                print(f"Total experiments run: {i}.")
        print("All experiments completed.")

    def load_experiments(self):
        if self.expt_names is None:
            self.expt_names = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        for expt_name in self.expt_names:
            expt_dir = self.root_dir / expt_name
            print(f"Loading experiment {expt_name} from {expt_dir}.")
            config = np.load(expt_dir / "config.npy", allow_pickle=True).item()
            self.experiments[expt_name] = Experiment(config=ExperimentConfig(**config))
            self.experiments[expt_name].losses = np.load(
                expt_dir / "losses.npy", allow_pickle=True
            ).item()
            self.experiments[expt_name].metrics = np.load(
                expt_dir / "metrics.npy", allow_pickle=True
            ).item()
            self.experiments[expt_name].logs = np.load(
                expt_dir / "logs.npy", allow_pickle=True
            ).item()

    def make_loss_plots(self, metric: str = "val_loss") -> List[Plot]:
        keys = list(visible_colors.values())
        batch_sizes = set(
            [expt.config.batch_size for expt in self.experiments.values()]
        )
        losses = set([expt.config.loss.value for expt in self.experiments.values()])
        opt_names = set(
            [expt.config.optimizer.value for expt in self.experiments.values()]
        )
        learning_rates = set(
            [expt.config.learning_rate for expt in self.experiments.values()]
        )
        momentums = set([expt.config.momentum for expt in self.experiments.values()])

        batch_sizes = sorted(batch_sizes)
        opt_names = sorted(opt_names)
        losses = sorted(losses)
        learning_rates = sorted(learning_rates)
        momentums = sorted(momentums)

        print(losses, batch_sizes, opt_names)

        new_row = False
        new_col = False
        plots = []
        completed_expts = []

        for l, loss in enumerate(losses):
            for b, bs in enumerate(batch_sizes):
                ind = 0
                for o, opt_name in enumerate(opt_names):
                    for expt_name, expt in self.experiments.items():
                        for lr, learning_rate in enumerate(learning_rates):
                            for mi, momentum in enumerate(momentums):
                                if (
                                    expt.config.batch_size == bs
                                    and expt.config.loss.value == loss
                                    and expt.config.optimizer.value == opt_name
                                    and expt_name not in completed_expts
                                    and expt.config.momentum == momentum
                                    and expt.config.learning_rate == learning_rate
                                ):
                                    print(f"Adding {expt_name} to plots.")

                                    val_metric = np.array(
                                        [m[metric] for m in expt.losses.values()]
                                    )
                                    ylabel = "".join([l[0] for l in loss.split("_")])

                                    if b == 0:
                                        ylabel = ylabel.upper()
                                    else:
                                        ylabel = ""

                                    if l == 0:
                                        title = f"Batch Size {bs}"
                                    else:
                                        title = ""

                                    if l == len(losses) - 1:
                                        xlabel = "Epoch"
                                    else:
                                        xlabel = ""

                                    if new_row:
                                        order = PlotOn.APPEND_DOWN
                                        new_col = False
                                        new_row = False

                                    elif new_col:
                                        order = PlotOn.APPEND_RIGHT
                                        new_col = False
                                        new_row = False
                                    else:
                                        order = PlotOn.RIGHT
                                        new_col = False
                                        new_row = False

                                    # row.append(Plot(X=np.arange(len(train_metric)), y=train_metric, plot_type=PlotType.LINE, plot_order=order, title=f'Train {loss}',legend=f"{opt_name}", color=keys[i]))
                                    legend = (
                                        f"{opt_name}_{learning_rate}_{momentum}"
                                        if opt_name
                                        in [
                                            Optimizer.MOMENTUM.value,
                                            Optimizer.NESTEROV.value,
                                        ]
                                        else f"{opt_name}_{learning_rate}"
                                    )
                                    plots.append(
                                        Plot(
                                            X=np.arange(len(val_metric)),
                                            title_size=10,
                                            y=val_metric,
                                            plot_type=PlotType.LINE,
                                            plot_order=order,
                                            ylabel=ylabel,
                                            xlabel=xlabel,
                                            title=title,
                                            legend=legend,
                                            color=keys[ind],
                                        )
                                    )
                                    ind += 1
                                    # i+=1
                                    completed_expts.append(expt_name)

                new_col = True
            new_row = True

        # only show title for first row of plots

        return plots

    def plot_best_and_worst_weight_flow(
        self,
        selected_loss: LossType = LossType.MSE,
        plot_type: PlotType = PlotType.SCATTER,
        learning_rate: float = 0.01,
    ) -> List[Plot]:
        colors = list(visible_colors.values())
        batch_sizes = set(
            [expt.config.batch_size for expt in self.experiments.values()]
        )
        losses = set(
            [
                expt.config.loss.value
                for expt in self.experiments.values()
                if expt.config.loss == selected_loss
            ]
        )
        opt_names = set(
            [expt.config.optimizer.value for expt in self.experiments.values()]
        )
        batch_sizes = sorted(batch_sizes)
        opt_names = sorted(opt_names)
        losses = sorted(losses)

        plots = []
        grad_plots = []

        # loop across all combinations and find best model for each loss function

        po = -1
        for o, opt_name in enumerate(opt_names):
            best_expt = None
            best_loss = np.inf
            worst_expt = None
            worst_loss = -np.inf

            best_weights_plots = []
            worst_weights_plots = []

            best_gradients_plots = []
            worst_gradients_plots = []

            bs = 0
            expt_name = None

            for l, loss in enumerate(losses):
                for b, bs in enumerate(batch_sizes):
                    for expt_name, expt in self.experiments.items():
                        if (
                            expt.config.batch_size == bs
                            and expt.config.loss.value == loss
                            and expt.config.optimizer.value == opt_name
                            and expt.config.learning_rate == learning_rate
                        ):
                            val_loss = np.array(
                                [m["val_loss"] for m in expt.losses.values()]
                            ).mean()
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_expt = expt_name
                            if val_loss > worst_loss:
                                worst_loss = val_loss
                                worst_expt = expt_name

            if expt_name is None:
                continue

            print(
                f"Best model for {selected_loss} with batch size {bs} and optimizer {opt_name}: {best_expt} with loss {best_loss}."
            )
            print(
                f"Worst model for {selected_loss} with batch size {bs} and optimizer {opt_name}: {worst_expt} with loss {worst_loss}."
            )

            best_expt = self.experiments[best_expt]
            worst_expt = self.experiments[worst_expt]
            epochs = np.array(best_expt.train_runner.logs["epochs"])
            best_weights = np.array(best_expt.train_runner.logs["weights"]).reshape(
                len(epochs), -1
            )
            best_biases = np.array(best_expt.train_runner.logs["biases"]).reshape(
                len(epochs), -1
            )
            best_weight_gradients = np.array(
                best_expt.train_runner.logs["weight_gradients"]
            ).reshape(len(epochs), -1)
            best_bias_gradients = np.array(
                best_expt.train_runner.logs["bias_gradients"]
            ).reshape(len(epochs), -1)

            worst_weights = np.array(worst_expt.train_runner.logs["weights"]).reshape(
                len(epochs), -1
            )
            worst_biases = np.array(worst_expt.train_runner.logs["biases"]).reshape(
                len(epochs), -1
            )
            worst_weight_gradients = np.array(
                worst_expt.train_runner.logs["weight_gradients"]
            ).reshape(len(epochs), -1)
            worst_bias_gradients = np.array(
                worst_expt.train_runner.logs["bias_gradients"]
            ).reshape(len(epochs), -1)

            real_weights = best_expt.real_weights
            real_biases = best_expt.real_biases
            real_weights_plots = []

            for widx in range(best_weights.shape[1]):
                best_weight = best_weights[:, widx].flatten()
                best_gradient = best_weight_gradients[:, widx].flatten()
                worst_weight = worst_weights[:, widx].flatten()
                worst_gradient = worst_weight_gradients[:, widx].flatten()
                color = colors[len(colors) % (widx + 1)]

                best_weights_plots.append(
                    Plot(
                        X=epochs,
                        y=best_weight,
                        plot_type=plot_type,
                        plot_order=PlotOn.RIGHT,
                        color=color,
                    )
                )
                worst_weights_plots.append(
                    Plot(
                        X=epochs,
                        y=worst_weight,
                        plot_type=plot_type,
                        plot_order=PlotOn.RIGHT,
                        color=color,
                    )
                )
                real_weights_plots.append(
                    Plot(
                        X=epochs,
                        y=real_weights[widx] * np.ones_like(epochs),
                        plot_type=PlotType.SCATTER,
                        plot_order=PlotOn.RIGHT,
                        color=color,
                        allow_animation=False,
                    )
                )

                best_gradients_plots.append(
                    Plot(
                        X=epochs,
                        y=best_gradient,
                        plot_type=plot_type,
                        plot_order=PlotOn.RIGHT,
                        color=color,
                    )
                )
                worst_gradients_plots.append(
                    Plot(
                        X=epochs,
                        y=worst_gradient,
                        plot_type=plot_type,
                        plot_order=PlotOn.RIGHT,
                        color=color,
                    )
                )

            bc = colors[len(colors) - 1]
            best_weights_plots.extend(
                [
                    Plot(
                        X=epochs,
                        y=[b.flatten() for b in best_biases],
                        plot_type=plot_type,
                        plot_order=PlotOn.RIGHT,
                        title_size=10,
                        title=f"Best Params ({best_expt.config.batch_size})",
                        color=bc,
                        marker="x",
                    ),
                ]
            )
            worst_weights_plots.extend(
                [
                    Plot(
                        X=epochs,
                        y=[b.flatten() for b in worst_biases],
                        plot_type=plot_type,
                        plot_order=PlotOn.RIGHT,
                        title_size=10,
                        title=f"Worst Params ({worst_expt.config.batch_size})",
                        color=bc,
                        marker="x",
                    ),
                ]
            )
            best_gradients_plots.extend(
                [
                    Plot(
                        X=epochs,
                        y=[bg.flatten() for bg in best_bias_gradients],
                        plot_type=plot_type,
                        plot_order=PlotOn.RIGHT,
                        title_size=10,
                        title=f"Best Params ({best_expt.config.batch_size})",
                        legend="Bias Gradients",
                        color=bc,
                        marker="x",
                    ),
                ]
            )
            worst_gradients_plots.extend(
                [
                    Plot(
                        X=epochs,
                        y=[bg.flatten() for bg in worst_bias_gradients],
                        plot_type=plot_type,
                        plot_order=PlotOn.RIGHT,
                        title_size=10,
                        title=f"Worst Params ({worst_expt.config.batch_size})",
                        legend="Bias Gradients",
                        color=bc,
                        marker="x",
                    ),
                ]
            )

            best_weights_plots[-1].legend = "Biases"
            best_weights_plots[0].legend = "Weights"

            worst_weights_plots[-1].legend = "Biases"
            worst_weights_plots[0].legend = "Weights"

            best_gradients_plots[0].legend = "Weight Gradients"
            worst_gradients_plots[0].legend = "Weight Gradients"

            real_weights_plots[-1].legend = "Real Weights"

            real_weights_plots[0].plot_order = PlotOn.APPEND_RIGHT
            real_weights_plots.append(
                Plot(
                    X=np.array([int(len(real_weights) // 2)]),
                    y=np.array([real_biases[0]]),
                    plot_type=plot_type,
                    plot_order=PlotOn.RIGHT,
                    legend="Real Bias",
                    linewidth=10,
                    title_size=10,
                    marker="+",
                    color=colors[len(colors) - 1],
                    allow_animation=False,
                )
            )

            worst_weights_plots[0].plot_order = PlotOn.APPEND_RIGHT
            worst_gradients_plots[0].plot_order = PlotOn.APPEND_RIGHT

            if po != o:
                best_weights_plots[0].plot_order = PlotOn.APPEND_DOWN
                best_weights_plots[-1].ylabel = opt_name
                best_gradients_plots[0].plot_order = PlotOn.APPEND_DOWN
                best_gradients_plots[-1].ylabel = opt_name
            if o == 0:
                # worst_weights_plots[-1].title='Worst Parameters'
                # best_weights_plots[-1].title='Best Parameters'
                real_weights_plots[-1].title = "Real Parameters"
            if o == len(opt_names) - 1:
                worst_weights_plots[-1].xlabel = "Epoch"
                best_weights_plots[-1].xlabel = "Epoch"
                best_gradients_plots[-1].xlabel = "Epoch"
                worst_gradients_plots[-1].xlabel = "Epoch"

                # real_weights_plots[0].xlabel='Epoch'

            grad_plots.extend(best_gradients_plots)
            grad_plots.extend(worst_gradients_plots)

            plots.extend(best_weights_plots)
            plots.extend(real_weights_plots)
            plots.extend(worst_weights_plots)
            po = o

        return plots, grad_plots

    def make_lr_plots(self, loss: LossType = LossType.MSE) -> List[Plot]:
        keys = list(visible_colors.values())
        batch_sizes = set(
            [expt.config.batch_size for expt in self.experiments.values()]
        )
        losses = set(
            [
                expt.config.loss.value
                for expt in self.experiments.values()
                if expt.config.loss == loss
            ]
        )
        opt_names = set(
            [expt.config.optimizer.value for expt in self.experiments.values()]
        )
        learning_rates = set(
            [expt.config.learning_rate for expt in self.experiments.values()]
        )
        momentums = set([expt.config.momentum for expt in self.experiments.values()])

        learning_rates = sorted(learning_rates)
        momentums = sorted(momentums)
        batch_sizes = sorted(batch_sizes)
        opt_names = sorted(opt_names)
        losses = sorted(losses)

        print(losses, batch_sizes, opt_names, learning_rates, momentums)

        new_row = False
        new_col = False
        plots = []
        completed_expts = []

        for l, lr in enumerate(learning_rates):
            for b, bs in enumerate(batch_sizes):
                cindex = 0
                for o, opt_name in enumerate(opt_names):
                    for mi, m in enumerate(momentums):
                        if mi > 0 and opt_name not in [
                            Optimizer.MOMENTUM.value,
                            Optimizer.NESTEROV.value,
                        ]:
                            continue
                        for expt_name, expt in self.experiments.items():
                            if (
                                expt.config.batch_size == bs
                                and expt.config.loss == loss
                                and expt.config.optimizer.value == opt_name
                                and expt.config.learning_rate == lr
                                and expt.config.momentum == m
                                and expt_name not in completed_expts
                            ):

                                print(f"Adding {expt_name} to plots.")

                                val_metric = np.array(
                                    [m["val_loss"] for m in expt.losses.values()]
                                )
                                ylabel = f"lr: {lr}"

                                if b == 0:
                                    ylabel = ylabel.upper()
                                else:
                                    ylabel = ""

                                if l == 0:
                                    title = f"Batch Size {bs}"
                                else:
                                    title = ""

                                if l == len(losses) - 1:
                                    xlabel = "Epoch"
                                else:
                                    xlabel = ""

                                if new_row:
                                    order = PlotOn.APPEND_DOWN
                                    new_col = False
                                    new_row = False

                                elif new_col:
                                    order = PlotOn.APPEND_RIGHT
                                    new_col = False
                                    new_row = False
                                else:
                                    order = PlotOn.RIGHT
                                    new_col = False
                                    new_row = False

                                # row.append(Plot(X=np.arange(len(train_metric)), y=train_metric, plot_type=PlotType.LINE, plot_order=order, title=f'Train {loss}',legend=f"{opt_name}", color=keys[i]))
                                legend = (
                                    f"{opt_name}"
                                    if opt_name
                                    not in [
                                        Optimizer.MOMENTUM.value,
                                        Optimizer.NESTEROV.value,
                                    ]
                                    else f"{opt_name}({m})"
                                )
                                plots.append(
                                    Plot(
                                        X=np.arange(len(val_metric)),
                                        title_size=10,
                                        y=val_metric,
                                        plot_type=PlotType.LINE,
                                        plot_order=order,
                                        ylabel=ylabel,
                                        xlabel=xlabel,
                                        title=title,
                                        legend=legend,
                                        color=keys[o + mi],
                                    )
                                )
                                # i+=1
                                completed_expts.append(expt_name)

                new_col = True
            new_row = True

        return plots


# expt_handler_config = ExperimentHandlerConfig(
#     root_dir=Path("expt_res"),
#     data_gen=[DataGenerator(DataGeneratorConfig(num_rows=1000, num_cols=5))],
#     losses=[LossType.MAE, LossType.MSE],
#     metrics=[LossType.MAE, LossType.MSE],
#     num_epochs=[100],
#     batch_sizes=[1, 16, 32, -1],
#     optimizers=[
#         Optimizer.SGD,
#         Optimizer.MOMENTUM,
#         Optimizer.NESTEROV,
#         Optimizer.ADAM,
#         Optimizer.RMSPROP,
#         Optimizer.ADAGRAD,
#         Optimizer.ADADELTA,
#         Optimizer.ADAMAX,
#     ],
#     seeds=[100],
#     learning_rates=[0.1, 0.01, 0.001, 0.0001],
#     momentums=[0.1, 0.9],
#     plot_format="pdf",
#     log_anim=False,
#     log_plots=True,
#     log_real_params=False,
# )
# expt_handler = ExperimentHandler(expt_handler_config.root_dir, expt_handler_config)
# expt_handler.make_experiments()
