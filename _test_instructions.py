"""
I have a python module called independent validation (IV). It figures out the
 accuracy of a machine learning classifier. Different to cross validation it
 returns not only a number but a probability distribution of the accuracy.
Your task is to write tests for my program. I hve already outlined what the tests
 should look like in the following. There is also a demo function on how to use IV.


weighted_sum_distribution(prob_dists, weights=None):
This function takes probability density distributions for random variables.
The prob_dists are scipy.stats distributions, subclasses of rv_continuous.
Let's call the random variables X, Y and Z and their corresponding distributions fx, fy, fz.
This function returns the distribution for a new random variable A = X + Y + Z.
fa is then the convolution of fx, fy and fz.
This function also takes weights for each random variable.
Let's call these weights wx, wy, wz.
If they are not all equal to 1 the formular changes.
The full formular is A = wx*X + wy*Y + wz*Z.
Multiplying a random variable B with a konstant k gives a new random variable C with C = B*k.
The probability density distribution is then:
f_C(c) = (1/|k|) * f_B(b/k)
This formular is used to get the probability density distribution for the weighted random variables.
Then the probability density distributions are convoluted to get a final distribution.
Test:
    - Test with 2 and 3 distributions
    - Test with different weights
        - test with all weights equal to 1 (assume they will be normalized by the function to 1/n_dists)
        - test with all weights equal and summing up to 1
        - test with unequal weights summing up to 1
    - Test that the result is accurate distribution at least once
    - Test that the means make sense always

MCMC
def metropolis_hastings(
        target_log_prob_fn,
        initial_value,
        num_samples,
        step_size=0.1,
        proposal_fn=None,
        burn_in=0,
        thin=1,
        random_seed=None
):
This function takes a target_log_prob_fn.
It then performs metropolis hastings mcmc.
It returns a tuple: samples, acceptance_rate
    - test with different possible target_log_prob_fn for which the target distribution is known.
    - check that resulting distribution is accurate


rv histogram subclass
I have a subclass of the scipy stats rv_histogram class called CustomHistogram(histogram_tuple=(hist, bin_edges)).
It should behave almost the same as the standard rv_histogram.
It has a method implemented to get the maximum a posteriori (MAP). object.map()
It has a method implemented called is_greater_than().
This method takes another instance of the class and then returns the probability that a random drawing from one instance is is bigger than one from the other instance.
    - check that it behaves the same as rv_histogram
    - check that the map is accurate in non arbitrary cases
    - check that the greater_than method is accurate in multiple cases

One func
def independent_validation(
        classifier,
        X,
        y,
        key="bacc",
        n=float("inf"),
        output="map",
        plot=False,
        iv_start_trainset_size=2,
        iv_batch_size=None,
        iv_n_batches=None,
        mcmc_num_samples=1000,
        mcmc_step_size=0.1,
        mcmc_burn_in=10000,
        mcmc_thin=50,
        mcmc_random_seed=None,
        asymptote_prior=None,
        offset_factor_prior=None,
):
This should always behave the same way as creating an IV object and then calling the run_iv method,
the compute_posterior method and then the get method.
    - check that it behaves the same as iv

IV
The IV is the core process. If something really doesn't work that would be visible in the results.
    - test run_iv method for edge cases
    - test compute_posterior for edge cases
    - test get method for all general cases
        - each type of key
        - different n values (5, 50, float('inf'))
        - with and without plot
        - with and without get_development (this then does not give a distribution but instead three lists that summarize how the distributions change over increasing n:
            - one list showing how the mean development over n
            - one list showing the lower bound of the distribution
            - one list showing the upper bound of the distribution


Test with different data types
    - demo with synthetic data
        - with 2 and with 3 classes
    - demo with wine data
    - use normal IV
    - use various combinations of parameters

Test content:
Create different datasets (gaussian, linear, non-linear) for which different classifiers work differently well.
Include one dataset where all classes are drawn from the same distribution, so that it is impossible to classify them well.
Based on theoretical considerations calculate the accuracy that each classifier should approach with enough data.
This is dependent on the dataset and on the classifier.
Run IV with the different classifiers on the datasets and check that the theoretical value and the accuracy fit.
Check this, by getting the interval of 95 percent of the distribution that is the result of IV
 and check that the theoretical value is within this interval.
Either way, also plot the distribution and the theoretical value in a graph.
Also get the accuracy distribution of a classifier on 5 samples, 15 samples and 50 samples. Plot all in the same graph.
Do this as well for the impossible dataset.
Plot the development of one classifier on 100 samples by setting get_development=True.
Plot the same thing for a classifier on the impossible dataset.


General Info:
Distributions are usually instances of scipy.stats.rv_continuous subclasses.
Most commonly it will be of the CustomHistogram class.
This is also what is returned by the get function of the IV object (unless get_development is True).

The get method of the iv class takes these arguments and returns an instance of CustomHistogram
def get(
        self,
        key, # Necessary: 'acc' for accuracy, 'bacc' for balanced accuracy or any label for the within label accuracy
        n=float('inf'),
        plot=False,
        get_development=False
):
"""
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from independent_validation.iv_file import IV
from independent_validation.weighted_sum_distribution import weighted_sum_distribution
from independent_validation.mcmc import metropolis_hastings
from independent_validation.rv_hist_subclass import CustomHistogram
from independent_validation.one_func import independent_validation

import pandas as pd


def demo_titanic_data():
    print("\n=== Demo: Real Data from Kaggle (Titanic Dataset) ===")
    try:
        # Attempt to load the Titanic dataset. (Place the file in a folder called "data".)
        data = pd.read_csv("data/train_data.csv")
    except Exception as e:
        print(
            "Could not load 'data/train_data.csv'. Please ensure the Kaggle Titanic dataset is present. Skipping real-data demo.")
        return

    # Preprocess the Titanic data.
    # Drop rows with missing values in key columns.
    cols_of_interest = ["Pclass_1" , "Pclass_2", "Pclass_3", "Sex", "Age", "Fare", "Survived"]
    data = data.dropna(subset=cols_of_interest).copy()


    # Use features: Pclass, Sex, Age, Fare and target: Survived.
    X = data[["Pclass_1" , "Pclass_2", "Pclass_3", "Sex", "Age", "Fare"]].values
    y = data["Survived"].values

    # Choose a classifier – here we use Logistic Regression.
    clf = LogisticRegression(max_iter=200)

    # Use the IV object to record the evolving performance.
    iv_real = IV(X, y, clf)
    iv_real.run_iv(start_trainset_size=10, batch_size=5)
    iv_real.compute_posterior(num_samples=1000, step_size=0.05, burn_in=100, thin=50, random_seed=42)
    dist_1 = iv_real.get(key=0, n=50, plot='Quicksave1')
    dist_2 = iv_real.get(key=1, n=50, plot='Quicksave2')
    means, lower_bounds, upper_bounds = iv_real.get_development(key=0, plot="Quicksave3")
    print("Done")

if __name__ == "__main__":
    print('Hi')
    demo_titanic_data()