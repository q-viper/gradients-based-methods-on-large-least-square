import streamsync as ss
from gradls.losses import LossType, Loss
from gradls.experiment import Experiment, DataGenerator, ExperimentConfig, Optimizer
from gradls.webapp.config import webapp_config

from dataclasses import dataclass
import json


def change_metrics_options(state: ss.State):
    metric = {k: k for k in state["loss_options"].keys() if k != state["loss"]}
    state["metrics_options"] = metric


def save_and_run_experiment(state: ss.State):
    curr_state = state.to_dict()
    expt_name = state["new_expt_name"].replace(" ", "_").lower()
    expt_dir = webapp_config.experiment_dir / f"experiment_{expt_name}.json"
    if expt_dir.exists() and ("false" == state["is_replace_old_expt"]):
        state.add_notification(
            "Error",
            f"Experiment {state['new_expt_name']} Already Exists",
            "Set replace it or choose a different name.",
        )
    else:
        # write curr_state to json file
        with open(str(expt_dir), "w") as f:
            json.dump(curr_state, f)
        print(f"Experiment {expt_name} saved!")
        state.add_notification(
            "Success",
            f"Experiment {state['new_expt_name']} Saved",
            "You can now run it.",
        )


# Shows in the log when the app starts
print("Hello world!")


initial_state = ss.init_state(
    {
        "my_app": {"title": "GradLS", "version": "0.1.0"},
        "new_expt_name": "My Experiment",
        "is_replace_old_expt": "true",
        "expt_num_rows": 1000,
        "expt_num_cols": 20,
        "expt_random_seed": 42,
        "min_x": 0,
        "max_x": 100,
        "expt_data_normalize": "true",
        "num_epochs": 100,
        "batch_size": 1,
        "learning_rate": 0.01,
        "optimizer_options": {k.value: k.value for k in Optimizer},
        "optimizer": Optimizer.SGD.value,
        "loss_options": {k.value: k.value for k in LossType},
        "loss": LossType.MSE.value,
        "metrics_options": {k.value: k.value for k in LossType},
        "metrics": [LossType.MAE.value, LossType.HINGE.value, LossType.RMSE.value],
        "log_anim": "true",
        "anim_fps": 10,
        "anim_frames": 10,
        "log_every": 1,
        "log_real_params": "true",
        "fig_nrows": 5,
        "fig_ncols": 10,
    }
)
