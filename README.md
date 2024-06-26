# Coding Tips for Cognitive Researchers

### Amric Trudel  
Laboratoire de Neurosciences Cognitives Computationnelles  
École Normale Supérieure  
Paris

![Comic strip on code quality](assets/comic_strip.png)

## Why using good coding practices?
- Code for your future self (in 2+ months)
- Reproduce any results that you generated in the past.
- Share your code and be easily understood by other researchers
- Someone in your lab can continue your work easily

----

## 1- Clean Code

### a) Variable and Function Naming
- Meaningful and pronounceable names
  - Variable: **noun** that describes the purpose of the variable. E.g.:   
    - `x` --> `observation`  
    - `sv` --> `sampling_variance`  
    - `b` --> `beta` --> `inverse_temperature`
  - Function: should contain a verb and express exactly what the function does. E.g.:  
    - `fit_model_parameters()`
    - `simulate_full_trajectory()`
    - `compute_psychophysical_kernel()`  
      
 
- Don't hesitate to split a calculation over many lines if you use longer variables.  
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

### b) Designing good functions
- Computing steps can be encapsulated into clearly named functions to make it understandable.
- Code that can be read like prose is preferable to writing comments (can you guess why?)
```python
def computre_and_plot_psychophysical_kernel(actions: np.ndarray, rewards: np.ndarray) -> None:
    features = extract_features(actions, rewards)
    target = extract_target(actions, rewards)
    coefficients, stderror = fit_logistic_regression(features, targets)
    plot_psychophysical_kernel(coefficients, stderrors)
```
- Ideally, your functions should:
  - Be small
  - Do one and ONLY ONE thing
  - Do exactly what their name indicates
  - Have no side effects
  
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

They are very useful in function signatures.
```python
def compute_action_repeating_probability(actions: np.ndarray, rewards: np.ndarray) -> float:
    probability: float = ...
    return probability
```

### d) Code design
- Keep configurability at high levels  
  In a file named **config.py** at the root of the repository.
  ```python
  DATA_DIR = 'data/'
  BATCH_SIZE = 1000
  MAX_REWARD = 100
  N_TRIALS = 80
  ```
- Magic numbers should be named constants (typically in bold characters)
- Be consistent across your code base
- Remove code instead of commenting it out and use **git** to find it back if you need it again.
- Don't hesitate to create classes for data formats that are outputted by your functions
  ```python
  class Trajectory:
    def __init__(self,
                 actions: np.ndarray,
                 rewards: np.ndarray,
                 description: Optional[str] = None
                 ):
        self.actions = actions
        self.rewards = rewards
        self.description: Optional[str] = description

    def __str__(self) -> str:
        return self.description

    def __len__(self) -> int:
        return self.actions.shape[1]
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

Many elements of your code can be conceptualized as objects:
- Model (with its parameters)
- Dataset (with its size, elements, saving location)
- Dataset Generator (with specific settings of volatility and stochasticity)
- Behavior analysis methods
- The environment itself, or the game (handles task dynamics)
- The players
- (eventually) the model trainer (with training configuration)

Simpler type objects:
- Trajectory
- Performance evaluation
- ... basically any time you have arrays and other quantities that need to be moved 
together from one function to another.

## 3 - Unit tests

A unit test verifies that a function has the desired behavior given a set of inputs. Developers write extensive unit tests
for their apps before they are deployed in production. In research, it might not be necessary to be as rigorous with the test coverage,
but unit tests can be useful to test critical calculations to make sure they do what is intended.

The `tests` folder should contain the tests and follow the same structure as the `cogvarlib` folder.
  
Unit test exercise: The `main` branch contains the solution. You can switch to the `demo` branch to try and write unit tests for the following methods of the RescorlaWagner class:
- `choose_action()`
- `update()`

## 4- Packaging your code
- Structure your code in modules (see the `cogvarlib` folder)
- The overall folder should hold the name of your library (some people also like to call it `src`)
- Your tests should go in a folder named `tests`
- You can put your notebooks at the root of the directory or, if you have many, in a `notebooks` folder
- Write your dependencies in `requirements.txt`
- Package your code as a library (`setup.py`)

## 4 - Using your code in Notebooks
- The whole point of packaging your code as a library is that you can then install it locally.  
  ```bash
  pip install -e .
  ```
- At the top of your notebook, add these magic commands to enable package auto-reload as you make changes:
    ```python
    %load_ext autoreload
    %autoreload 2
    ```
- Import your packages in your notebook  
  ```python
  from cogvarlib.models import RescorlaWagner
  ```

## 5- Versioning with Git

First, you need to understand that git tacks the **changes** that you make to your code, NOT the actual versions of the code.
You can structure your history with a series of **commits** that encompass a series of changes, and the succession of commits
form a **branch**.

### Basic commands

Saving your code changes to git is a three-step process
- Add the files that contain changes you want to register:  
`git add <files_to_add>`  
- Commit your changes to a snapshot of your code that you might eventually return to:  
`git commit -m "<your commit message>`  
- Push your code to github.com (optional):  
`git push`

Other commands:
- See the state of your uncommited changes:  
`git status`
- Create a branch:  
`git checkout -b <name_of_your_new_branch>`
- Change branches:
`git checkout <name_of_the_branch>`
- Merge your branch to the main branch:
  ```bash
  git checkout main
  git merge <branch_to_merge>
  ```
- Create a tag:
`git tag <name_of_the_tag>`
- Go to the commit associated with a tag:
`git checkout <name_of_the_tag`


### Tips
- Create a `.gitignore` file where you list all files that should not be saved by git.  
  (This is SUPER important for files that contain API and database credentials.)
- You can create aliases: 
`git config --global alias.hist log --all --decorate --oneline --graph`  
The following is useful to visualize the history of your commits, by typing `git hist`
- Download [Oh my zsh](https://ohmyz.sh/) in order to pip up your shell and always be informed of
what branch you are on.

### Useful coding patterns
- Structure your work in clearly-named commits
- Document your code evolution (no need to keep multiple copies of the same code)
- Use tags:
  - Keep track of the code version that is associated with each experiment you did
- Branch:
  - Mainly if you work with other people
  - Can be useful if you are trying something out. !! But merge it quickly !!
  - Branches are not meant to contain alternative versions of your code. They are meant to facilitate concurrent development of different features without interference. But the purpose is to have a main branch onto which all evolutions are merged.

### More resources
- https://gitimmersion.com
- https://learngitbranching.js.org (interactive, amazing if you want to understand branches)

## 6- Experiments
Your script that launches an experiment, training, or analysis can automate routine jobs:
- Ensuring you have committed the code being executed
- Automatically number the experiment
- Create a tag on the commit
- Save the results

Come and see me for more details :)

## 7- Data processing
Think about data processing as a Pipeline, with multiple steps.

A Pipeline has:
- Input file location
- Output directory
- Series of steps

A Step has:
- A CLEAR name
- A **standard** input format 
- A **standard** output format

Notes:
- Intermediary formats should also have a clear name.
- Steps are put one after the other in the pipeline.
- Intermediary results can also be saved if they are long to compute
