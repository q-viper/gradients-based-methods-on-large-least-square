from gradls.vis.colors import visible_colors
from gradls.vis.matplotlib_vis import (
    Plot,
    PlotOn,
    PlotType,
    MatplotlibVizConfig,
    MatplotlibVisualizer,
)
from gradls.losses import Loss, LossType
from gradls.datagenerator import DataGenerator, MyDataset

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import numpy as np
import torch.nn as nn
from typing import List
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
from enum import Enum


class Optimizer(Enum):
    SGD = "sgd"
    MOMENTUM = "momentum"
    NESTEROV = "nesterov"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAMAX = "adamax"


@dataclass
class ExperimentConfig:
    name: str
    viz_config: MatplotlibVizConfig
    num_epochs: int
    batch_size: int
    loss: LossType
    model: Optional[nn.Module] = None
    optimizer: Optimizer = Optimizer.SGD
    learning_rate: float = 0.01
    momentum: float = 0.0
    train_valid_split: float = 0.1
    seed: Optional[int] = 100
    metrics: List[LossType] = field(default_factory=lambda: [LossType.MAE])
    log_every: int = 1
    log_dir: Path = None
    log_anim: bool = True
    log_plots: bool = True
    plot_format: str = "png"
    anim_interval: int = 100
    anim_frames: int = 10
    anim_fps: int = 30
    log_real_params: bool = True
    log_real_data: bool = True
    verbose: bool = False
    device: str = "cpu"


class Runner:
    def __init__(
        self,
        name: str,
        model: nn.Module,
        batch_size: int,
        optimizer: str,
        loss: Loss,
        metrics: Optional[List[Loss]] = [],
        log_every: int = 1,
        log_params: bool = False,
        is_test: bool = False,
        data: Dataset = None,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        device: str = "cuda",
    ):
        self.name = name
        self.batch_size = batch_size
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.is_test = is_test
        self.data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.logs = {metric.name: [] for metric in self.metrics}
        self.logs[f"{name}_loss({loss.name})"] = []
        self.curr_epoch = 0
        self.log_every = log_every
        self.log_params = log_params
        self.logs["epochs"] = []
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.device = device
        self.model.to(self.device)

        if log_params:
            self.logs["weights"] = []
            self.logs["biases"] = []
            self.logs["weight_gradients"] = []
            self.logs["bias_gradients"] = []
            self.logs["learning_rate"] = []

    def step(self):
        if self.is_test:
            self.model.eval()
        else:
            self.model.train()

        batch_losses = []
        batch_metrics = {metric.name: [] for metric in self.metrics}
        # print(self.metrics)

        # all_preds = []

        for i, (X, y) in enumerate(self.data_loader):
            # print(X.shape, y.shape)
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)

            # if self.log_output:
            #     all_preds.extend(y_pred.detach().numpy().tolist())

            loss = self.loss(y, y_pred.squeeze())
            batch_losses.append(loss.item())

            params = torch.cat([p.view(-1) for p in self.model.parameters()])
            # print(loss)

            if self.l1_penalty > 0:
                loss += self.l1_penalty * torch.abs(params).sum()
            if self.l2_penalty > 0:
                loss += self.l2_penalty * (params**2).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for metric in self.metrics:
                batch_metrics[metric.name].append(metric(y, y_pred).detach().numpy())

        if self.curr_epoch % self.log_every == 0:
            epoch_loss = np.mean(batch_losses)
            epoch_metrics = {
                metric.name: np.mean(batch_metrics[metric.name])
                for metric in self.metrics
            }
            self.logs[f"{self.name}_loss({self.loss.name})"].append(epoch_loss)
            for metric in self.metrics:
                self.logs[metric.name].append(epoch_metrics[metric.name])

            self.logs["epochs"].append(self.curr_epoch)

            if self.log_params:
                self.logs["weights"].append(self.model.weight.data.numpy().copy())
                self.logs["biases"].append(self.model.bias.data.numpy().copy())
                self.logs["weight_gradients"].append(
                    self.model.weight.grad.data.numpy().copy()
                )
                self.logs["bias_gradients"].append(
                    self.model.bias.grad.data.numpy().copy()
                )
                self.logs["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
        self.curr_epoch += 1

        # if self.log_output:
        #     self.predictions.append(all_preds)

        return epoch_loss, epoch_metrics


class Experiment:
    def __init__(self, config: ExperimentConfig):
        if config.seed is None:
            config.seed = np.random.randint(0, 100)
        torch.manual_seed(config.seed)
        self.config = config
        self.viz_config = config.viz_config
        self.viz = MatplotlibVisualizer(config=self.viz_config)
        self.losses = {}
        self.metrics = {metric.value: [] for metric in self.config.metrics}
        self.optimizer = None
        self.real_weights = None
        self.real_biases = None

        self.loss_fxn = Loss(self.config.loss)

    def load_data(self, data: DataGenerator):
        self.data_config = data.config
        self.data = data
        X, y = data.make_data()
        self.real_weights = data.weights
        self.real_biases = data.biases

        if self.config.model is None:
            self.config.model = nn.Linear(in_features=X.shape[1], out_features=1)

        train_X, val_X, train_y, val_y = train_test_split(
            X, y, test_size=self.config.train_valid_split, random_state=self.config.seed
        )
        self.train_X, self.val_X, self.train_y, self.val_y = (
            train_X,
            val_X,
            train_y,
            val_y,
        )
        self.make_runners()

    def make_runners(self):
        train_data = MyDataset(self.train_X, self.train_y)
        val_data = MyDataset(self.val_X, self.val_y)

        if self.config.optimizer == Optimizer.SGD:
            self.optimizer = torch.optim.SGD(
                self.config.model.parameters(), lr=self.config.learning_rate
            )
        elif self.config.optimizer == Optimizer.MOMENTUM:
            self.optimizer = torch.optim.SGD(
                self.config.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
            )
        elif self.config.optimizer == Optimizer.NESTEROV:
            self.optimizer = torch.optim.SGD(
                self.config.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                nesterov=True,
            )
        elif self.config.optimizer == Optimizer.ADAM:
            self.optimizer = torch.optim.Adam(
                self.config.model.parameters(), lr=self.config.learning_rate
            )
        elif self.config.optimizer == Optimizer.RMSPROP:
            self.optimizer = torch.optim.RMSprop(
                self.config.model.parameters(), lr=self.config.learning_rate
            )
        elif self.config.optimizer == Optimizer.ADAGRAD:
            self.optimizer = torch.optim.Adagrad(
                self.config.model.parameters(), lr=self.config.learning_rate
            )
        elif self.config.optimizer == Optimizer.ADADELTA:
            self.optimizer = torch.optim.Adadelta(
                self.config.model.parameters(), lr=self.config.learning_rate
            )
        elif self.config.optimizer == Optimizer.ADAMAX:
            self.optimizer = torch.optim.Adamax(
                self.config.model.parameters(), lr=self.config.learning_rate
            )
        elif self.config.optimizer == Optimizer.ADAMW:
            self.optimizer = torch.optim.AdamW(
                self.config.model.parameters(), lr=self.config.learning_rate
            )
        else:
            raise ValueError("Invalid optimizer type.")

        if self.config.batch_size < 1:
            self.config.batch_size = len(train_data)
            train_batch_size = len(train_data)
            test_batch_size = len(val_data)
        else:
            train_batch_size = self.config.batch_size
            test_batch_size = self.config.batch_size

        self.train_runner = Runner(
            name="train",
            loss=self.loss_fxn,
            model=self.config.model,
            batch_size=train_batch_size,
            optimizer=self.optimizer,
            metrics=[Loss(m) for m in self.config.metrics],
            data=train_data,
            log_params=True,
            device=self.config.device,
        )
        self.val_runner = Runner(
            name="val",
            loss=self.loss_fxn,
            model=self.config.model,
            batch_size=test_batch_size,
            optimizer=self.optimizer,
            metrics=[Loss(m) for m in self.config.metrics],
            is_test=True,
            data=val_data,
            device=self.config.device,
        )

    def train(self):
        for epoch in range(self.config.num_epochs):
            train_loss, train_metrics = self.train_runner.step()
            val_loss, val_metrics = self.val_runner.step()
            self.losses[epoch] = {"train_loss": train_loss, "val_loss": val_loss}
            for metric in self.config.metrics:
                self.metrics[metric.value].append(
                    {
                        "train": train_metrics[metric.value],
                        "val": val_metrics[metric.value],
                    }
                )

            # train_mse, val_mse, train_bias, val_bias, train_variance, val_variance = self.calculate_bias_variance()

            if epoch % self.config.log_every == 0 and self.config.verbose:
                print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}")

        print("Training complete.")

    def get_default_plots(self, plot_parameters: bool = True) -> List[Plot]:
        plots = []
        colors = visible_colors
        keys = list(colors.values())
        # np.random.shuffle(keys)
        train_loss = np.array([m["train_loss"] for m in self.losses.values()])
        val_loss = np.array([m["val_loss"] for m in self.losses.values()])
        plots.append(
            Plot(
                X=np.arange(self.config.num_epochs),
                y=train_loss,
                plot_type=PlotType.LINE,
                plot_order=PlotOn.RIGHT,
                legend="train_loss",
                color=keys[0],
            )
        )
        plots.append(
            Plot(
                X=np.arange(self.config.num_epochs),
                y=val_loss,
                plot_type=PlotType.LINE,
                plot_order=PlotOn.RIGHT,
                legend="val_loss",
                color=keys[1],
            )
        )

        for metric in self.config.metrics:
            train_metric = np.array([m["train"] for m in self.metrics[metric.value]])
            val_metric = np.array([m["val"] for m in self.metrics[metric.value]])

            plots.append(
                Plot(
                    X=np.arange(len(train_metric)),
                    y=train_metric,
                    plot_type=PlotType.LINE,
                    plot_order=PlotOn.APPEND_RIGHT,
                    legend=f"train_{metric.value}",
                    color=keys[2],
                )
            )
            plots.append(
                Plot(
                    X=np.arange(len(val_metric)),
                    y=val_metric,
                    plot_type=PlotType.LINE,
                    plot_order=PlotOn.RIGHT,
                    legend=f"val_{metric.value}",
                    color=keys[3],
                )
            )

        if plot_parameters and self.train_runner.log_params:
            epochs = np.array(self.train_runner.logs["epochs"])
            weights = np.array(self.train_runner.logs["weights"]).reshape(
                len(epochs), -1
            )
            biases = np.array(self.train_runner.logs["biases"]).reshape(len(epochs), -1)
            weight_gradients = np.array(
                self.train_runner.logs["weight_gradients"]
            ).reshape(len(epochs), -1)
            bias_gradients = np.array(self.train_runner.logs["bias_gradients"]).reshape(
                len(epochs), -1
            )

            weights_plots = []
            biases_plots = []
            gradients_plots = []
            real_weights = self.real_weights

            for widx in range(weights.shape[1]):
                weight = weights[:, widx].flatten()
                weight_gradient = weight_gradients[:, widx].flatten()
                color = keys[len(colors) % (widx + 1)]
                if widx == 0:
                    weights_plots.append(
                        Plot(
                            X=epochs,
                            y=weight,
                            plot_type=PlotType.SCATTER,
                            plot_order=PlotOn.APPEND_DOWN,
                            title="Weights",
                            color=color,
                        )
                    )

                    gradients_plots.append(
                        Plot(
                            X=epochs,
                            y=weight_gradient,
                            plot_type=PlotType.SCATTER,
                            plot_order=PlotOn.APPEND_RIGHT,
                            title="Gradients",
                            color=color,
                        )
                    )
                else:
                    weights_plots.append(
                        Plot(
                            X=epochs,
                            y=weight,
                            plot_type=PlotType.SCATTER,
                            plot_order=PlotOn.RIGHT,
                            color=color,
                            title="Weights",
                            xlabel="Epoch",
                        )
                    )

                    gradients_plots.append(
                        Plot(
                            X=epochs,
                            y=weight_gradient,
                            plot_type=PlotType.SCATTER,
                            plot_order=PlotOn.RIGHT,
                            color=color,
                            xlabel="Epoch",
                            title="Gradients",
                        )
                    )

            bc = keys[len(colors) - 1]
            weights_plots.extend(
                [
                    Plot(
                        X=epochs,
                        y=[b.flatten() for b in biases],
                        plot_type=PlotType.SCATTER,
                        plot_order=PlotOn.RIGHT,
                        marker="x",
                        title="Trained Param.",
                        xlabel="Epoch",
                        color=bc,
                        legend="Trained Bias",
                    ),
                    #  Plot(X=epochs, y=[self.real_biases]*np.ones_like(epochs), plot_type=PlotType.SCATTER, plot_order=PlotOn.RIGHT, marker='x',
                    #  title='Biases', xlabel='Epoch', legend='Real Bias', color=bc, allow_animation=False)
                ]
            )
            gradients_plots.extend(
                [
                    Plot(
                        X=epochs,
                        y=[bg.flatten() for bg in bias_gradients],
                        plot_type=PlotType.SCATTER,
                        plot_order=PlotOn.RIGHT,
                        marker="x",
                        title="Param. Gradients",
                        xlabel="Epoch",
                        color=bc,
                        legend="Bias Gradients",
                    ),
                    #  Plot(X=epochs, y=[0]*np.ones_like(epochs), plot_type=PlotType.SCATTER, plot_order=PlotOn.RIGHT, marker='x',
                    #  title='Biases', xlabel='Epoch', legend='Real Bias', color=bc, allow_animation=False)
                ]
            )

            # if self.config.log_real_params:
            #     weights_plots[-1].legend='Real Weights'
            weights_plots[-2].legend = "Trained Weights"
            weights_plots[-1].xlabel = "Epoch"

            gradients_plots[-2].legend = "Weight Gradients"

            plots.extend(weights_plots)
            plots.extend(biases_plots)
            plots.extend(gradients_plots)
            if self.config.log_real_params:
                plots.extend(
                    [
                        Plot(
                            X=np.arange(len(real_weights)),
                            y=real_weights,
                            plot_type=PlotType.SCATTER,
                            plot_order=PlotOn.APPEND_RIGHT,
                            title="Real Parameters",
                            legend="Real Weights",
                            color=keys[4],
                            allow_animation=False,
                        ),
                        Plot(
                            X=np.array([int(len(real_weights) // 2)]),
                            y=np.array([self.real_biases[0]]),
                            plot_type=PlotType.SCATTER,
                            plot_order=PlotOn.RIGHT,
                            legend="Real Bias",
                            linewidth=10,
                            marker="+",
                            color=keys[10],
                            allow_animation=False,
                        ),
                    ]
                )

        print(f"Found {len(plots)} metrics to plot.")
        return plots

    def animate_plots(self, interval: int = 10, frames: int = 10):
        return self.viz.animate_plots(interval, frames)

    def save_fig(self, fig, path: Path, format: str = "svg"):
        self.viz.save_fig(fig, path, format)

    def show_fig(self):
        self.viz.show_fig()

    def log_expt(self, plots: Optional[List[Plot]] = None) -> Path:

        if self.config.log_dir is None:
            self.config.log_dir = Path("expt_res")
        expt_dir = Path(f"{self.config.log_dir}/{self.config.name}")
        expt_dir.mkdir(parents=True, exist_ok=True)

        if self.config.log_plots:
            if plots is None:
                plots = self.get_default_plots()
            self.viz.clear_plots()
            for plot in plots:
                self.viz.append_plot(plot)
            fig, ax = self.viz.generate_plots()
            self.save_fig(
                fig,
                Path(f"{expt_dir}/{self.config.name}.{self.config.plot_format}"),
                format=self.config.plot_format,
            )
        if self.config.log_anim:
            self.animate_plots(
                interval=self.config.anim_interval, frames=self.config.anim_frames
            ).save(
                Path(f"{expt_dir}/{self.config.name}.gif"),
                writer="imagemagick",
                fps=self.config.anim_fps,
            )

        # log expt config
        expt_config = asdict(self.config)
        np.save(
            Path(f"{self.config.log_dir}/{self.config.name}/config.npy"), expt_config
        )

        print(f"Experiment results saved at {expt_dir}.")

        if self.config.log_real_data:
            # store train and val data as numpy files
            np.save(Path(f"{expt_dir}/train_X.npy"), self.train_X.numpy())
            np.save(Path(f"{expt_dir}/train_y.npy"), self.train_y.numpy())
            np.save(Path(f"{expt_dir}/val_X.npy"), self.val_X.numpy())
            np.save(Path(f"{expt_dir}/val_y.npy"), self.val_y.numpy())
            print("Data saved.")

        # store all logs from runners
        logs = {
            key: runner.logs
            for key, runner in zip(
                ["train", "val"], [self.train_runner, self.val_runner]
            )
        }
        np.save(Path(f"{expt_dir}/logs.npy"), logs)
        print("Logs saved.")

        if self.config.log_real_params:
            # store real weights and biases
            np.save(Path(f"{expt_dir}/real_weights.npy"), self.real_weights)
            np.save(Path(f"{expt_dir}/real_biases.npy"), self.real_biases)
            print("Real weights and biases saved.")
        # store all params from model, no need its in experiment config
        # torch.save(self.config.model.state_dict(), Path(f"{expt_dir}/model.pth"))
        # print("Model saved.")

        # store all losses and metrics
        np.save(Path(f"{expt_dir}/losses.npy"), self.losses)
        np.save(Path(f"{expt_dir}/metrics.npy"), self.metrics)
        print("Losses and metrics saved.")
        return expt_dir


# in_features = 20
# out_features = 1
# viz_config = MatplotlibVizConfig(figsize=(15,10),title="My Exp",
#                                  use_tex=False)
# exp_config = ExperimentConfig(name="Exp5", loss=LossType.MAE, viz_config=viz_config,
#                               num_epochs=10, batch_size=24,learning_rate=0.1, optimizer=Optimizer.ADAM,
#                               model=None, metrics=[LossType.MSE, LossType.RMSE],
#                               log_every=1, log_anim=False, anim_fps=10, plot_format='png')

# data = DataGenerator(DataGeneratorConfig(num_rows=1000, num_cols=in_features, min_val=0, max_val=100, seed=100, normalize=True, noise=0.1))

# exp = Experiment(config=exp_config)
# exp.load_data(data)
# exp.train()
