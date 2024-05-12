from pathlib import Path

from dataclasses import dataclass


@dataclass
class WebAppConfig:
    experiment_dir: Path = Path("experiments")
    look_every: int = 10


webapp_config = WebAppConfig()
