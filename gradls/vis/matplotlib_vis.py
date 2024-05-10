import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from pydantic import dataclasses
from enum import Enum
from matplotlib.animation import FuncAnimation


@dataclasses.dataclass
class MatplotlibVizConfig:
    # config for scientific visualization with matplotlib and save as svg
    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 300
    background_color: str = "white"
    font_size: int = 20
    font_family: str = "Arial"
    font_color: str = "black"
    linewidth: int = 2
    num_rows: int = 1
    num_cols: int = 1


class PlotOn(Enum):
    APPEND_RIGHT = "append_right"
    APPEND_DOWN = "append_down"
    RIGHT = "right"


class PlotType(Enum):
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"


@dataclasses.dataclass
class Plot:
    X: np.ndarray
    y: Optional[np.ndarray]
    plot_type: PlotType
    plot_order: PlotOn


class MatplotlibVisualizer:
    def __init__(self, config: MatplotlibVizConfig):
        self.config = config
        self._plots = []

    def make_figure(self):
        fig, ax = plt.subplots(
            self.config.num_rows,
            self.config.num_cols,
            figsize=self.config.figsize,
            dpi=self.config.dpi,
            facecolor=self.config.background_color,
        )
        return fig, ax

    def append_plot(
        self,
        plot: Plot,
    ):
        self._plots.append(plot)

    def _get_num_rows_cols(self) -> Tuple[int, int]:
        num_rows = 1
        num_cols = 1
        for plot in self._plots:
            if plot.plot_order == PlotOn.APPEND_DOWN:
                num_rows += 1
            elif plot.plot_order == PlotOn.APPEND_RIGHT:
                num_cols += 1
        return num_rows, num_cols

    def generate_plots(self):
        # get rows and cols number
        num_rows, num_cols = self._get_num_rows_cols()
        # get figure and axes
        fig, ax = self.make_figure()

        curr_row, curr_col = 0, 0
        curr_ax = ax[curr_row, curr_col]

        for plot in self._plots:
            if plot.plot_order == PlotOn.APPEND_DOWN:
                curr_row += 1
            elif plot.plot_order == PlotOn.APPEND_RIGHT:
                curr_col += 1

            curr_ax = ax[curr_row, curr_col]
            if plot.plot_type == PlotType.LINE:
                curr_ax.plot(plot.X, plot.y)
            elif plot.plot_type == PlotType.SCATTER:
                curr_ax.scatter(plot.X, plot.y)
            elif plot.plot_type == PlotType.BAR:
                curr_ax.bar(plot.X, plot.y)

        return fig, ax

    def save_fig(self, fig, path: Path):
        fig.savefig(path, format="svg", dpi=self.config.dpi)

    def show_fig(self, fig):
        plt.show()

    def close_fig(self, fig):
        plt.close(fig)
