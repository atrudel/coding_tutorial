# Coding Tips for Researchers

## Why using good coding practices?
- Code for your future self (in 2+ months)
- Reproduce any results that you generated in the past.
- Share your code and be easily understood by other researchers
- Someone in your lab can continue your work easily

## 1- Clean Code

### a) Variable and Function Naming
- Meaningful and pronounceable names
  - Variable: **noun** that describes the purpose of the variable 
    `x` --> `observation`
  - Function: verbs
* Don't hesitate to split a calculation over many lines if you use longer variables   
Example: Rescorla-Wagner model
```python
q = 0.5
a = 0.7

# Update rw model
q = q + a * (r - q)
```
Can become:
```python
q_value = 0.5
learning_rate = 0.7

def update_rescorla_wagner_model(reward):
    temporal_difference = reward - q_value
    q_value += learning_rate * temporal_difference
```
Note: Updating variables outside the function is not recommended, but we will see later how to use object-oriented programming to make it cleaner.

### b) Encapsulation
- Computing steps can be encapsulated into clearly named functions to make it understandable.
- Code that can be read like prose is preferable to comments (can you guess why?)
```python
def psychophysical_kernel(actions: np.ndarray, rewards: np.ndarray) ->None:
    features = extract_features(actions, rewards)
    target = extract_target(actions, rewards)
    coefficients, stderror = fit_logistic_regression(features, targets)
    plot_psychophysical_kernel(coefficients, stderrors)
```
Each of the subfunctions called in `psychophysical_kernel` can then be defined individually.
### c) Type hinting
Although python is not a strongly typed language, unlike C or java, we can use type hinting in order to allow our IDE to highlight our errors.
- Native types
```python
learning_rate: float = 0.6
action: int = 1
plot_result: bool = True
```
- Types from other libraries
```python
import numpy as np
import torch

trajectory: np.ndarray = np.array([0, 1, 2, 3, 4, 5])
batch: torch.Tensor = torch.Tensor([[0.95, 0.1, 0.34], [0.34, 0.0, 0.2]])
```
- Compound types
```python
from typing import Dict, List, Tuple

parameters: Dict[str, float] = {'learning_rate': 0.7, 'temperature': 0.9}
trials: List[Tuple[int, float]] = [(0, 0.5), (1, 0.2), (0, 0.6)]  # Trials as tuples (action, reward)
```

## 2- Object-Oriented Programming

In Python you can write **classes** to create sections of code that act like **objects**.
```python
import numpy as np
from typing import Tuple

class RescorlaWagner:
    def __init__(self, learning_rate, temperature):
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.q_values = np.array([0.5, 0.5])

    def choose_action(self) -> Tuple[int, float]:
        prob = 1 / (1 + np.exp(-self.temperature * (self.q_values[1] - self.q_values[0])))
        action = np.random.binomial(1, prob)
        return action, prob

    def update(self, last_action: int, last_reward: float) -> None:
        td_error = last_reward - self.q_values[last_action]
        self.q_values[last_action] += self.learning_rate * td_error
```
In this example, we wrote a class for the Rescorla-Wagner model for a binary action.  
- The parameters of the model are written as **attributes** and can be passed to the **constructor** when we instantiate a model:
  ```python
  rw_model = RescorlaWagner(learning_rate=0.7, temperature=0.9)
  ```
- The class has two **methods** that can be called on the instantiated model
  ```python
  action = rw_model.choose_action()
  ```
  And then given a reward:
  ```python
  rw_model.update(action, reward)
  ```

- Data Loader

## 3 - Unit tests

## 4 - Using your code in Notebooks
- Structure your code
- Package your code as a library (`setup.py`)
- `pip install -e .`
- Add magic commands to enable package auto-reload in the notebook:
    ```python
    %load_ext autoreload
    %autoreload 2
    ```

## 5- Versioning with Git
### Basic commands

### Tips
- Create a `.gitignore` file

### Useful patterns
- Structure your work
- Document your code evolution (no need to )
- Use tags:
  - Model versions
- Branch:
  - Mainly if you work with other people
  - Can be useful if you are trying something out. !! But merge it quickly !!
  - Branches are not meant to contain alternative versions of your code. They are meant to facilitate concurrent development of different features without interference. But the purpose is to have a main branch onto which all evolutions are merged.

## 6- Experiments
- Experiment running script

## 7- Data processing

