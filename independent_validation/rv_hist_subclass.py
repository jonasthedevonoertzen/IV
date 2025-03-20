import numpy as np
from scipy import stats

from independent_validation.weighted_sum_distribution import weighted_sum_distribution


class CustomHistogram(stats.rv_histogram):
    """
    Custom histogram distribution with additional functionality for
    MAP estimation and probability comparison.
    """

    def __init__(self, histogram_tuple, *args, **kwargs):
        """
        Initialize the custom histogram distribution.

        Parameters:
        -----------
        histogram_tuple : tuple
            Tuple containing (hist, bin_edges) as returned by np.histogram
        """
        super().__init__(histogram_tuple, *args, **kwargs)
        self.hist, self.bin_edges = histogram_tuple

    def map_estimate(self):
        """
        Calculate the Maximum A Posteriori (MAP) estimate.

        Returns:
        --------
        float
            The x value corresponding to the highest density point
        """
        # Find the bin with maximum count
        max_bin_idx = np.argmax(self.hist)
        # Return the midpoint of that bin
        return (self.bin_edges[max_bin_idx] + self.bin_edges[max_bin_idx + 1]) / 2


    def is_greater_than(self, other_dist):
        """
        Compare if this distribution's CDF is greater than another distribution's CDF at point x.

        Parameters:
        -----------
        other_dist : rv_continuous or rv_discrete
            The other distribution to compare with
        x : float
            The point at which to compare CDFs

        Returns:
        --------
        bool
            True if this distribution's CDF is greater at point x
        """
        if not isinstance(other_dist, CustomHistogram):
            res = 1 - self.cdf(other_dist)
            return res
        prob_greater_than = weighted_sum_distribution([self, other_dist], weights=[1, -1], normalize_weights=False)
        res = 1 - prob_greater_than.cdf(0.)
        return res

    def map(self):
        return self.map_estimate()

    # Signifikanzniveau, soll p wert zurückgeben - ppf

    # confidence interval - interval