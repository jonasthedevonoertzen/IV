# IV: Independent Validation

IV is a Python package designed to assess the accuracy of machine learning classifiers by generating a full probability distribution of their performance. This method goes beyond traditional cross-validation by using an iterative process and Markov Chain Monte Carlo (MCMC) to estimate uncertainty.
By only using samples for training after they have been tested alpha inflation is prevented.

## Features

- **Independent Validation:** Gradually expands the training set while recording prediction outcomes.
- **Posterior Sampling:** Uses Metropolis-Hastings to compute posterior distributions of accuracy.
- **Custom Distribution Handling:** Provides a custom histogram subclass for MAP estimation and distribution comparison.
- **Simple Interface:** Use the one-function wrapper (`independent_validation`) for quick experiments.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd IV
   ```

2 **Install dependencies:**
    ```bash
    pip install numpy scipy scikit-learn matplotlib pandas pytest
    ```
## Usage

An example of running the independent validation process with a classifier:

```python
from independent_validation.one_func import independent_validation
from sklearn.linear_model import LogisticRegression
import numpy as np

# Generate or load your dataset
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

# Run independent validation
result = independent_validation(
    classifier=LogisticRegression(max_iter=200),
    X=X,
    y=y,
    key="acc",  # Use "acc" for accuracy or "bacc" for balanced accuracy
    n=50,
    output='map',
    plot=False,
    mcmc_num_samples=1000,
    mcmc_step_size=0.1,
    mcmc_burn_in=10000,
    mcmc_thin=50,
    mcmc_random_seed=42
)

print("MAP Accuracy:", result)
```
## Project Structure

- independent_validation/
    - iv_file.py — Core iterative validation process.
    - mcmc.py — Implements the Metropolis-Hastings MCMC sampler.
    - one_func.py — One-function interface for independent validation.
    - rv_hist_subclass.py — Custom histogram with extended functionality.
    - weighted_sum_distribution.py — Utility for combining probability distributions.
- Tests & Demos:
    - combined_file.txt — Contains test descriptions and instructions.
    - _test_instructions.py — Provides detailed test cases and a demo using real data.

## License

MIT License
