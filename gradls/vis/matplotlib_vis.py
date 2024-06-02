import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from matplotlib.animation import FuncAnimation


@dataclass
class MatplotlibVizConfig:
    # config for scientific visualization with matplotlib and save as svg
    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 100
    background_color: str = "white"
    font_size: int = 20
    font_family: str = "Arial"
    font_color: str = "black"
    linewidth: int = 2
    show_grid: bool = True
    use_tex: bool = True
    legend_loc: str = "upper right"
    title: str = ""
    hide_empty_plots: bool = True


class PlotOn(Enum):
    APPEND_RIGHT = "append_right"
    APPEND_DOWN = "append_down"
    RIGHT = "right"


class PlotType(Enum):
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    HIST = "hist"
    IMAGE = "image"


@dataclass
class Plot:
    X: np.ndarray
    y: Optional[np.ndarray]
    plot_type: PlotType
    plot_order: PlotOn
    plot_row_col: Tuple[int, int] = (-1, -1)
    title: str = ""
    title_size: int = 15
    xlabel: str = ""
    ylabel: str = ""
    label_size: int = 10
    color: str = "blue"
    marker: Optional[str] = None
    linestyle: str = "-"
    linewidth: int = 2
    legend: Optional[str] = "_nolegend_"
    show_legend: bool = True
    legend_size: int = 7
    apply_margin: bool = False
    top_margin: float = 0.9
    bottom_margin: float = 0.1
    left_margin: float = 0.1
    right_margin: float = 0.9
    xmax: Optional[float] = None
    xmin: Optional[float] = None
    ymax: Optional[float] = None
    ymin: Optional[float] = None
    allow_animation: bool = True


class MatplotlibVisualizer:
    def __init__(self, config: MatplotlibVizConfig):
        self.config = config
        self._plots = []
        self.curr_row = 0
        self.curr_col = 0
        self.num_rows = 1
        self.num_cols = 1
        self.num_plots = 0
        self.ax_lim = {}
        self.completed_frames = []

    def make_figure(self):
        print(f"Making figure with {self.num_rows} rows and {self.num_cols} columns.")
        fig, ax = plt.subplots(
            self.num_rows,
            self.num_cols,
            figsize=self.config.figsize,
            dpi=self.config.dpi,
            facecolor=self.config.background_color,
        )
        fig.suptitle(
            self.config.title,
            fontsize=self.config.font_size,
            fontfamily=self.config.font_family,
            color=self.config.font_color,
        )
        plt.tight_layout()
        # use latex on text
        if self.config.use_tex:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
        return fig, np.array(ax).reshape(self.num_rows, self.num_cols)

    def append_plot(
        self,
        plot: Plot,
        left_to_right: bool = True,
    ):
        if self.num_plots == 0:
            self.num_plots += 1
        else:
            if plot.plot_order == PlotOn.APPEND_RIGHT:
                if left_to_right:
                    self.curr_col += 1
            elif plot.plot_order == PlotOn.APPEND_DOWN:
                self.curr_row += 1
                if left_to_right:
                    self.curr_col = 0
            elif plot.plot_order == PlotOn.RIGHT:
                pass
            else:
                raise ValueError("Invalid plot order.")
            self.num_plots += 1
            self.num_rows = max(self.num_rows, self.curr_row + 1)
            self.num_cols = max(self.num_cols, self.curr_col + 1)

        plot.plot_row_col = (self.curr_row, self.curr_col)

        if plot.X is not None:
            if plot.xmax is None:
                plot.xmax = np.nanmax(plot.X)
            if plot.xmin is None:
                plot.xmin = np.nanmin(plot.X)
            if plot.plot_type == PlotType.IMAGE:
                plot.xmax = plot.X.shape[1]
                plot.xmin = 0
        if plot.y is not None:
            if plot.ymax is None:
                plot.ymax = np.nanmax(plot.y)
            if plot.ymin is None:
                plot.ymin = np.nanmin(plot.y)

        self._plots.append(plot)

    def _generate_plot(self, plot: Plot, ax, limit: int = -1):
        curr_row, curr_col = plot.plot_row_col
        key = f"{curr_row}_{curr_col}"
        label = plot.legend
        # print(f"Shape of X: {plot.X.shape}, Limit: {limit}, Label: {label}.")
        if plot.plot_type == PlotType.LINE:
            ax.plot(
                plot.X[:limit],
                plot.y[:limit],
                color=plot.color,
                marker=plot.marker,
                linestyle=plot.linestyle,
                linewidth=plot.linewidth,
                label=label,
            )
        elif plot.plot_type == PlotType.SCATTER:
            ax.scatter(
                plot.X[:limit],
                plot.y[:limit],
                color=plot.color,
                marker=plot.marker,
                linewidth=plot.linewidth,
                label=label,
            )
        elif plot.plot_type == PlotType.BAR:
            ax.bar(plot.X[:limit], plot.y[:limit], color=plot.color, label=label)
        elif plot.plot_type == PlotType.HIST:
            ax.hist(plot.X[:limit], color=plot.color, label=label)
        elif plot.plot_type == PlotType.IMAGE:
            ax.imshow(plot.X)
        else:
            raise ValueError("Invalid plot type.")

        # add left margin
        # print(ax.get_xlim(), ax.get_ylim())
        x_min, x_max = plot.xmin, plot.xmax  # ax.get_xlim()
        y_min, y_max = plot.ymin, plot.ymax  # ax.get_ylim()

        if self.ax_lim.get(key) is None:
            self.ax_lim[key] = []
        else:
            # print(f"Existing limits: {self.ax_lim[key]}, keys: {key}")
            x_min = (
                min(x_min, self.ax_lim[key][0])
                if (self.ax_lim[key][0] is not None and x_max is not None)
                else ax.get_xlim()[0]
            )
            x_max = (
                max(x_max, self.ax_lim[key][1])
                if (self.ax_lim[key][1] is not None and x_max is not None)
                else ax.get_xlim()[1]
            )
            y_min = (
                min(y_min, self.ax_lim[key][2])
                if (self.ax_lim[key][2] is not None and y_min is not None)
                else ax.get_ylim()[0]
            )
            y_max = (
                max(y_max, self.ax_lim[key][3])
                if (self.ax_lim[key][3] is not None and y_max is not None)
                else ax.get_ylim()[1]
            )
        self.ax_lim[key] = [x_min, x_max, y_min, y_max]

        if plot.apply_margin:
            ax.set_xlim([x_min - plot.left_margin, x_max + plot.right_margin])
            if plot.y is not None:
                ax.set_ylim([y_min - plot.bottom_margin, y_max + plot.top_margin])

        ax.set_title(
            plot.title,
            fontsize=plot.title_size,
            fontfamily=self.config.font_family,
            color=self.config.font_color,
        )
        ax.set_xlabel(
            plot.xlabel,
            fontsize=plot.label_size,
            fontfamily=self.config.font_family,
            color=self.config.font_color,
        )
        ax.set_ylabel(
            plot.ylabel,
            fontsize=plot.label_size,
            fontfamily=self.config.font_family,
            color=self.config.font_color,
        )
        ax.grid(self.config.show_grid)

        return ax

    def generate_plots(self, save_path: Optional[Path] = None, format: str = "svg"):
        # get figure and axes
        fig, ax = self.make_figure()

        for plot in self._plots:
            curr_row, curr_col = plot.plot_row_col
            curr_ax = ax[curr_row, curr_col]

            curr_ax = self._generate_plot(plot, curr_ax, limit=plot.X.shape[0])

        for plot in self._plots:
            curr_row, curr_col = plot.plot_row_col
            curr_ax = ax[curr_row, curr_col]
            if plot.show_legend:
                curr_ax.legend(loc=self.config.legend_loc, fontsize=plot.legend_size)

        # hide empty plots
        if self.config.hide_empty_plots:
            plot_row_col = [plot.plot_row_col for plot in self._plots]
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    if (i, j) not in plot_row_col:
                        ax[i, j].axis("off")

        if save_path is not None:
            self.save_fig(fig, save_path, format=format)

        return fig, ax

    def _animate(self, curr_frame, ax, num_frames=30):
        curr_frame += 1
        print(
            f"Animating frame {curr_frame}. Completed frames: {self.completed_frames}. Num frames: {num_frames}."
        )

        for plot in self._plots:
            anim_plot = True
            curr_row, curr_col = plot.plot_row_col
            curr_ax = ax[curr_row, curr_col]
            if curr_frame > 1 or curr_frame in self.completed_frames:
                plot.legend = "_nolegend_"

            if plot.allow_animation:
                limit = int((curr_frame + 1) * len(plot.X) / num_frames)
            else:
                if curr_frame == 1:
                    limit = len(plot.X)
                else:
                    anim_plot = False
            if anim_plot:
                curr_ax = self._generate_plot(plot, curr_ax, limit=limit)
        self.completed_frames.append(curr_frame)

        for plot in self._plots:
            curr_row, curr_col = plot.plot_row_col
            curr_ax = ax[curr_row, curr_col]
            if plot.show_legend:
                curr_ax.legend(loc=self.config.legend_loc, fontsize=plot.legend_size)

        # hide empty plots
        if self.config.hide_empty_plots:
            plot_row_col = [plot.plot_row_col for plot in self._plots]
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    if (i, j) not in plot_row_col:
                        ax[i, j].axis("off")

        return ax

    def animate_plots(
        self, interval: int = 10, frames: int = 10, save_path: Optional[Path] = None
    ):
        fig, ax = self.make_figure()
        anim = FuncAnimation(
            fig,
            self._animate,
            frames=frames,
            fargs=(
                ax,
                frames,
            ),
            interval=interval,
        )
        if save_path is not None:
            self.save_animation(anim, save_path)
        return anim

    def save_animation(self, anim, path: Path):
        anim.save(path)

    def save_fig(self, fig, path: Path, format: str = "svg"):
        fig.savefig(path, format=format, dpi=self.config.dpi)

    def show_fig(self):
        plt.show()

    def close_fig(self, fig):
        plt.close(fig)

    def clear_plots(self):
        self._plots = []
        self.num_plots = 0
        self.curr_row = 0
        self.curr_col = 0
        self.num_rows = 1
        self.num_cols = 1
        plt.clf()


# Example usage
# viz_config = MatplotlibVizConfig(figsize=(5,3),title="Test",
#                                  use_tex=False,)
# viz = MatplotlibVisualizer(config=viz_config)

# X = np.linspace(0, 10, 100)
# y = np.sin(X)
# z = np.cos(X)
# w = y+z

# # show legend in latex format

# plot1 = Plot(X=X, y=y, plot_type=PlotType.LINE, plot_order=PlotOn.RIGHT, legend="sin(x)")
# plot2 = Plot(X=X, y=y, plot_type=PlotType.SCATTER, plot_order=PlotOn.APPEND_RIGHT)
# plot3 = Plot(X=X, y=y, plot_type=PlotType.BAR, plot_order=PlotOn.APPEND_DOWN)
# plot4 = Plot(X=X, y=z, plot_type=PlotType.LINE, plot_order=PlotOn.APPEND_RIGHT, legend="cos(x)", color="green", allow_animation=False)
# plot5 = Plot(X=np.array([1]*100), y=w, plot_type=PlotType.SCATTER, plot_order=PlotOn.RIGHT, legend="sin(x)+cos(x)", color="red")
# plot5 = Plot(X=np.array([10]*100), y=w, plot_type=PlotType.SCATTER, plot_order=PlotOn.RIGHT, legend="", color="red", allow_animation=False)

# plot6 = Plot(X=np.random.randint(0, 10, 100), y=None, plot_type=PlotType.HIST, plot_order=PlotOn.APPEND_DOWN)
# plot7 = Plot(X=np.random.randint(0, 255,(10,10, 3)), y=None, plot_type=PlotType.IMAGE, plot_order=PlotOn.APPEND_RIGHT, title="Image")


# viz.append_plot(plot1)
# viz.append_plot(plot2)
# viz.append_plot(plot3)
# viz.append_plot(plot4)
# viz.append_plot(plot5)
# viz.append_plot(plot6)
# viz.append_plot(plot7)


# fig, ax = viz.generate_plots()
# viz.save_fig(fig, Path("expt_res/test.png"), 'png')
# viz.show_fig()
# viz.animate_plots(interval=100,frames=10).save('expt_res/test.gif', writer='imagemagick', fps=30)
