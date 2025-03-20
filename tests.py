"""
test_independent_validation.py

This file provides unit and integration tests for the independent validation (IV)
package. It covers the following functions/methods:

1. weighted_sum_distribution:
    - Test with 2 and 3 distributions.
    - Test with equal default weights (assumed to be normalized to 1/n).
    - Test with explicit weights summing to 1 (both equal and unequal).
    - Test that the resulting distribution has the expected mean.

2. metropolis_hastings:
    - Test sampling from a known 1D target distribution (e.g. standard normal) and
      verify that the resulting samples have a mean and standard deviation close
      to the target's.

3. CustomHistogram (rv_histogram subclass):
    - Compare the pdf from CustomHistogram to the standard rv_histogram on several points.
    - Check that the MAP method returns the bin center corresponding to the maximum density.
    - Check that the is_greater_than method gives results in the expected direction.

4. independent_validation (one_func):
    - Test that running the wrapper function provides the same outcome as using IV directly.

5. IV (core object):
    - Test run_iv and compute_posterior methods on small datasets (edge cases).
    - Test on synthetic data (both 2-class and 3-class).
    - Test on an “impossible” dataset (features and labels random) where accuracy
      should approach chance level.

Some tests also exercise the plotting or "development" functionality in a way that
only numerical outcomes (like means) are checked.
"""

import numpy as np
import pytest
from scipy.stats import norm, rv_histogram
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Import functions/classes from the independent validation package.
from independent_validation.weighted_sum_distribution import weighted_sum_distribution
from independent_validation.mcmc import metropolis_hastings
from independent_validation.rv_hist_subclass import CustomHistogram
from independent_validation.iv_file import IV
from independent_validation.one_func import independent_validation


#############################
# weighted_sum_distribution #
#############################

def test_weighted_sum_distribution_equal_weights_two():
    """
    Test weighted_sum_distribution with two distributions.
    When no weights are provided, they should be normalized to 1/n.
    For two distributions: expected mean = (mean1 + mean2)/2.
    """
    # Create two normal distributions
    d1 = norm(loc=0, scale=1)
    d2 = norm(loc=1, scale=2)
    combined_dist = weighted_sum_distribution([d1, d2])
    expected_mean = (d1.mean() + d2.mean()) / 2.0

    # Use the built-in expect method to compute the mean of the combined distribution
    mean_val = combined_dist.expect()
    assert np.allclose(mean_val, expected_mean, atol=0.1)


def test_weighted_sum_distribution_equal_weights_three():
    """
    Test weighted_sum_distribution with three distributions with default weights.
    Expected mean = average of individual means.
    """
    d1 = norm(loc=0, scale=1)
    d2 = norm(loc=1, scale=2)
    d3 = norm(loc=2, scale=1.5)
    combined_dist = weighted_sum_distribution([d1, d2, d3])
    expected_mean = (d1.mean() + d2.mean() + d3.mean()) / 3.0

    mean_val = combined_dist.expect()
    assert np.allclose(mean_val, expected_mean, atol=0.1)


def test_weighted_sum_distribution_explicit_equal_weights():
    """
    Test weighted_sum_distribution with two distributions and explicit weights all set to 1.
    The function is assumed to normalize these weights to (1/2, 1/2).
    """
    d1 = norm(loc=0, scale=1)
    d2 = norm(loc=10, scale=2)
    combined_dist = weighted_sum_distribution([d1, d2], weights=[1, 1])
    expected_mean = (d1.mean() + d2.mean()) / 2.0

    mean_val = combined_dist.expect()
    assert np.allclose(mean_val, expected_mean, atol=0.5)


def test_weighted_sum_distribution_unequal_weights():
    """
    Test weighted_sum_distribution with two distributions and unequal weights
    that sum to 1. For example, weights [0.3, 0.7] should give mean = 0.3*d1.mean + 0.7*d2.mean.
    """
    d1 = norm(loc=0, scale=1)
    d2 = norm(loc=10, scale=2)
    weights = [0.3, 0.7]
    combined_dist = weighted_sum_distribution([d1, d2], weights=weights)
    expected_mean = weights[0] * d1.mean() + weights[1] * d2.mean()

    mean_val = combined_dist.expect()
    assert np.allclose(mean_val, expected_mean, atol=0.5)


####################
# metropolis_hastings #
####################

def test_metropolis_hastings_normal():
    """
    Test the metropolis_hastings function on a 1D standard normal distribution.
    Check that the sample mean and standard deviation are close to 0 and 1 respectively.
    """
    target_log_prob_fn = lambda x: norm.logpdf(x)
    samples, acceptance_rate = metropolis_hastings(
        target_log_prob_fn,
        initial_value=0.0,
        num_samples=10000,
        step_size=0.5,
        burn_in=100,
        thin=1,
        random_seed=42,
    )
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)
    assert np.allclose(sample_mean, 0, atol=0.1)
    assert np.allclose(sample_std, 1, atol=0.1)


###########################
# CustomHistogram testing #
###########################

def test_custom_histogram_behaviour():
    """
    Verify that a CustomHistogram instance behaves similarly to a standard rv_histogram.
    Compare pdf values over a set of points.
    """
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    hist, bin_edges = np.histogram(data, bins=50, density=True)
    custom_hist = CustomHistogram((hist, bin_edges))
    standard_hist = rv_histogram((hist, bin_edges))

    xs = np.linspace(-3, 3, 10)
    for x in xs:
        pdf_custom = custom_hist.pdf(x)
        pdf_standard = standard_hist.pdf(x)
        assert np.allclose(pdf_custom, pdf_standard, atol=1e-2)


def test_custom_histogram_map():
    """
    Test that the maximum a posteriori (MAP) value returned by CustomHistogram.map()
    is at (or very near) the center of the bin having the highest density.
    """
    rng = np.random.default_rng(123)
    data = rng.normal(0, 1, 1000)
    hist, bin_edges = np.histogram(data, bins=50, density=True)
    custom_hist = CustomHistogram((hist, bin_edges))

    idx_max = np.argmax(hist)
    bin_center = 0.5 * (bin_edges[idx_max] + bin_edges[idx_max + 1])
    map_val = custom_hist.map()

    # Allow for a tolerance roughly equal to the bin width
    tol = (bin_edges[1] - bin_edges[0])
    assert np.allclose(map_val, bin_center, atol=tol)


def test_custom_histogram_is_greater_than():
    """
    Test the is_greater_than method in a scenario where one distribution
    (centered near 0) is likely to produce lower values than another (centered near 1).
    """
    rng = np.random.default_rng(456)
    data1 = rng.normal(0, 1, 1000)
    data2 = rng.normal(1, 1, 1000)
    hist1, bins1 = np.histogram(data1, bins=50, density=True)
    hist2, bins2 = np.histogram(data2, bins=50, density=True)

    hist_obj1 = CustomHistogram((hist1, bins1))
    hist_obj2 = CustomHistogram((hist2, bins2))

    prob = hist_obj1.is_greater_than(hist_obj2)

    # Because hist_obj1 is centered lower than hist_obj2, the probability
    # that a draw from hist_obj1 is bigger should be below 0.5.
    assert prob < 0.5


###############################
# independent_validation (one_func)
###############################

def test_independent_validation_wrapper():
    """
    Test that the one_func `independent_validation` behaves the same as using IV directly.
    Do this by running both approaches on a synthetic dataset and comparing
    a statistic (here, the expected value of the resulting accuracy distribution).
    """
    # Create synthetic binary classification data.
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    clf = LogisticRegression(max_iter=200)

    # Use the one_func wrapper.
    dist_wrapper = independent_validation(
        classifier=clf,
        X=X,
        y=y,
        output='dist',
        mcmc_num_samples=500,
        mcmc_burn_in=50,
        mcmc_thin=5,
        plot=False,
    )

    # Now, do the same using the IV class.
    iv = IV(X, y, clf)
    iv.run_iv(start_trainset_size=2, batch_size=10)
    iv.compute_posterior(num_samples=500, burn_in=50, thin=5)
    dist_manual = iv.get(n=50, key='acc')

    # Compare the two distributions via their means.
    mean_wrapper = dist_wrapper.expect()
    mean_manual = dist_manual.expect()
    assert np.allclose(mean_wrapper, mean_manual, atol=0.2)


############################
# IV core functionality tests
############################

def test_iv_run_iv_edge_cases():
    """
    Test IV.run_iv on a small dataset to check that it works on edge cases.
    """
    X, y = make_classification(n_samples=10, n_features=3, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
    clf = LogisticRegression(max_iter=100)
    iv = IV(X, y, clf)
    try:
        iv.run_iv(start_trainset_size=2, batch_size=1)
    except Exception as e:
        pytest.fail(f"IV.run_iv failed on edge case with small training set: {e}")

    # Check that a distribution can be produced without error.
    dist = iv.get(key='acc', n=5)
    assert dist is not None


def test_iv_compute_posterior_edge_cases():
    """
    Test IV.compute_posterior on a small dataset to check that it handles edge cases.
    """
    X, y = make_classification(n_samples=20, n_features=4, n_classes=2, random_state=42)
    clf = LogisticRegression(max_iter=100)
    iv = IV(X, y, clf)
    iv.run_iv(start_trainset_size=2, batch_size=2)
    try:
        iv.compute_posterior(num_samples=100, burn_in=50, thin=10)
    except Exception as e:
        pytest.fail(f"IV.compute_posterior failed on edge case: {e}")


def test_iv_synthetic_data_two_classes():
    """
    Test IV on synthetic 2-class data.
    The classifier should achieve accuracy better than chance (0.5).
    """
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
    clf = LogisticRegression(max_iter=200)
    iv = IV(X, y, clf)
    iv.run_iv(start_trainset_size=5, batch_size=5)
    iv.compute_posterior(num_samples=500, burn_in=50, thin=5)
    dist = iv.get(key='acc', n=50)
    mean_accuracy = dist.expect()
    assert mean_accuracy > 0.5 + 0.1


def test_iv_synthetic_data_three_classes():
    """
    Test IV on synthetic 3-class data.
    For 3 classes, chance is ~0.33 so classifier accuracy should be noticeably higher.
    """
    X, y = make_classification(
        n_samples=300, n_features=5, n_classes=3, n_informative=3, random_state=42
    )
    clf = LogisticRegression(max_iter=200, multi_class="multinomial")
    iv = IV(X, y, clf)
    iv.run_iv(start_trainset_size=5, batch_size=5)
    iv.compute_posterior(num_samples=500, burn_in=50, thin=5)
    dist = iv.get(key='acc', n=50)
    mean_accuracy = dist.expect()
    # Expect accuracy to be above chance for multiclass (~0.33)
    assert mean_accuracy > 0.33 + 0.1


def test_iv_impossible_dataset():
    """
    Test IV on an "impossible" dataset where features are random and have no relation to labels.
    In binary classification, the classifier should perform at chance (around 0.5).
    """
    np.random.seed(24)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, size=100)
    clf = LogisticRegression(max_iter=200)
    iv = IV(X, y, clf)
    iv.run_iv(start_trainset_size=15, batch_size=5)
    iv.compute_posterior(num_samples=500, burn_in=500, thin=100)
    dist = iv.get(key='acc', n=50)
    mean_accuracy = dist.expect()
    assert np.abs(mean_accuracy - 0.5) < 0.1


def test_iv_get_development():
    """
    Test the get_development method that is supposed to return the development of the classifier
    as training progresses. Check that the returned tuple has consistent lengths.
    """
    X, y = make_classification(n_samples=150, n_features=5, n_classes=2, random_state=42)
    clf = LogisticRegression(max_iter=200)
    iv = IV(X, y, clf)
    iv.run_iv(start_trainset_size=5, batch_size=5)
    # Assume plot=False (or provide a dummy file name) to avoid actual file generation during testing.
    results = iv.get_development(key='acc', plot=False)
    # Expecting three arrays: means, lower_bounds, upper_bounds.
    assert len(results) == 3
    means, lower_bounds, upper_bounds = results
    assert len(means) == len(lower_bounds) == len(upper_bounds)


# To allow running tests directly.
if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
