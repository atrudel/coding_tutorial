import numpy as np

from cogvarlib.models.rescorla_wagner import RescorlaWagner


def test_rescorla_wagner_update():
    # Given
    lr = 0.8
    model = RescorlaWagner(learning_rate=lr, temperature=1, decay_rate=0)
    action = 0
    reward = 0.4

    # When
    model.update(action, reward)

    # Then
    expected_q0 = 0.5 + lr * (reward - 0.5)
    assert model.q_values[0] == expected_q0

def test_rescorla_wagner_decay():
    # Given
    lr = 0.8
    temp = 1.0
    decay = 0.2
    model = RescorlaWagner(learning_rate=lr, temperature=1, decay_rate=decay)
    model.q_values = np.array([0.7, 0.7])
    action = 0
    reward = 0.4

    # When
    model.update(action, reward)

    # Then
    expected_q1 = 0.7 + decay * (0.5 - 0.7)
    assert model.q_values[1] == expected_q1

