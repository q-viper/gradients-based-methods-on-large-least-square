from typing import Union, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass


@dataclass
class DataGeneratorConfig:
    num_rows: int
    num_cols: int
    min_val: int = 0
    max_val: int = 100
    seed: int = 100
    normalize: bool = True
    noise: Optional[Union[float, np.ndarray]] = None
    weights: Optional[np.ndarray] = None
    biases: Optional[np.ndarray] = None
    noise_norm_by: Optional[float] = 1.0


class DataGenerator:
    def __init__(self, config: DataGeneratorConfig):
        self.config = config

        np.random.seed(self.config.seed)
        self.X = np.random.uniform(
            self.config.min_val,
            self.config.max_val + 1,
            (self.config.num_rows, self.config.num_cols),
        )
        self.noise = (
            self.config.noise
            if self.config.noise is not None
            else np.random.standard_normal(self.config.num_rows)
            / self.config.noise_norm_by
        )
        self.normalize = self.config.normalize
        if self.config.normalize:
            self.X = self.X / np.max(self.X)
            # self.noise = self.noise/self.config.max_val
        self.weights = self.config.weights
        self.biases = self.config.biases
        # self.noise = self.config.noise if self.config.noise is not None else np.random.randn(self.config.num_rows)

    def make_data(self):
        if self.weights is None:
            self.weights = np.random.standard_normal(self.X.shape[1])
        if self.biases is None:
            self.biases = np.random.standard_normal(1)

        self.y = np.dot(self.X, self.weights) + self.biases + self.noise
        self.X, self.y = torch.tensor(self.X, dtype=torch.float32), torch.tensor(
            self.y, dtype=torch.float32
        )
        return self.X, self.y


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            raise ValueError("No y values found. Please generate data first.")
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    ## make sure the data is generated correctly
    data_gen1 = DataGenerator(DataGeneratorConfig(num_rows=1000, num_cols=20))
    data_gen1.make_data()

    data_gen2 = DataGenerator(DataGeneratorConfig(num_rows=1000, num_cols=20))
    data_gen2.make_data()
    assert data_gen2.weights == data_gen1.weights, data_gen2.biases == data_gen1.biases
