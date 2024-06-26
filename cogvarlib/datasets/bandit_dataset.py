from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class BanditDataset(Dataset):
    def __init__(self, means: np.ndarray, values: np.ndarray):
        self.means: np.ndarray = means.astype('float32')
        self.values: np.ndarray = values.astype('float32')

    @classmethod
    def load(cls, filename: str, directory: str = 'data') -> BanditDataset:
        data = np.load(Path(directory) / filename)
        means = data[:, :, :, 0].astype('float32')
        values = data[:, :, :, 1].astype('float32')
        return cls(means, values)

    def subset(self, size: int) -> BanditDataset:
        return BanditDataset(
            self.means[:size],
            self.values[:size],
        )

    @property
    def n_trials(self):
        return self.values.shape[2]

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, item: int):
        means = self.means[item]
        values = self.values[item]
        target = np.argmax(values, axis=0).astype('float32')
        return means, values, target

    def plot(self, item: int):
        means, values, target = self[item]
        plt.plot(values[0], label="Bandit 0: reward values", color="tab:blue")
        plt.plot(values[1], label="Bandit 1: reward values", color="orange")
        plt.plot(means[0], label="Bandit 0: latent mean", color="tab:cyan", linestyle="dotted")
        plt.plot(means[1], label="Bandit 1: latent mean", color="tan", linestyle="dotted")
        plt.scatter(list(range(len(values[0]))), target, label="target")
        plt.title(f"Bandit trajectories (no {item})")
        plt.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
        plt.xlabel("Time step")
        plt.ylabel("Reward")

