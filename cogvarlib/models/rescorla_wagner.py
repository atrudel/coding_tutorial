import numpy as np
from typing import Tuple


BASELINE = 0.5


class RescorlaWagner:
    def __init__(self, learning_rate, temperature, decay_rate):
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.decay_rate = decay_rate
        self.q_values = np.array([BASELINE, BASELINE])

    def choose_action(self) -> Tuple[int, float]:
        prob = 1 / (1 + np.exp(-self.temperature * (self.q_values[1] - self.q_values[0])))
        action = np.random.binomial(1, prob)
        return action, prob

    def update(self, last_action: int, last_reward: float) -> None:
        td_error = last_reward - self.q_values[last_action]
        self.q_values[last_action] += self.learning_rate * td_error
        self._decay(1 - last_action)

    def _decay(self, unchosen_action: int) -> None:
        self.q_values[unchosen_action] += self.decay_rate * (BASELINE - self.q_values[unchosen_action])

