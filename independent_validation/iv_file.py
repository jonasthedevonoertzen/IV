import warnings
import numpy as np
import math

from sklearn.base import clone
from independent_validation.mcmc import metropolis_hastings
from independent_validation.weighted_sum_distribution import weighted_sum_distribution
from independent_validation.rv_hist_subclass import CustomHistogram


class IV:
    def __init__(self, x_data, y_data, classifier):
        self.asymptotes = None
        self.offset_factors = None

        # Shuffle the dataset (x_data and y_data remain aligned)
        indices = np.random.permutation(len(x_data))

        # Save data and classifier (a clone is stored to avoid external modifications)
        self.x_data = x_data[indices]
        self.y_data = y_data[indices]
        self.classifier = clone(classifier)

        # The unique labels in the data.
        self._labels = np.unique(y_data)

        # For each label, record lists of:
        #   'sizes': the training set size at prediction time
        #   'outcomes': 1 if prediction correct, 0 otherwise.
        self._iv_records = {label: {'sizes': [], 'outcomes': []} for label in self._labels}

        # To hold posterior MCMC samples for each label: {label: (samples, acceptance_rate)}
        self._posterior = {}

        # Caches for distributions
        self._accuracy_cache = {}  # key: (label, n)
        self._bacc_cache = {}      # key: n
        self._acc_cache = {}       # key: n
        self._multi_cache = {}     # key: (key, n)
        self._development_cache = {}  # key: key -> dict

        # Frequency for each label in the full dataset for weighted overall accuracy.
        total = len(y_data)
        self._label_frequencies = {}
        for label in self._labels:
            self._label_frequencies[label] = np.sum(y_data == label) / total

        self.development_confidence_range = None

    def run_iv(self, start_trainset_size=2, batch_size=1):
        """
        Runs the Independent Validation (IV) process.

        The process:
          1. The classifier is initially trained on the first 'start_trainset_size' samples.
          2. For the remaining samples, in batches of 'batch_size':
             - The current classifier predicts the label(s) for the new sample(s).
             - For each sample, the outcome (1 for a correct prediction; 0 otherwise)
               is recorded along with the current training set size.
             - The new sample(s) are then added to the training set and the classifier is retrained.

        The outcomes are stored in self.iv_records, organized by true label.
        """
        # TODO: Include time estimator xx
        n_total = len(self.x_data)
        if start_trainset_size >= n_total:
            raise ValueError("start_trainset_size must be less than the total number of samples.")

        # Initial training set: use the first start_trainset_size samples.
        train_indices = list(range(start_trainset_size))

        # Process remaining samples in order, batch-by-batch.
        current_index = start_trainset_size

        # Extend until all classes are inside trainingsset
        while len(np.unique(self.y_data[train_indices])) < len(self._labels) and current_index < n_total:
            batch_indices = list(range(current_index, min(current_index + batch_size, n_total)))
            x_batch = self.x_data[batch_indices]
            y_batch = self.y_data[batch_indices]
            current_train_size = len(train_indices)
            # guessing prediction
            predictions = np.random.choice(self._labels, size=len(batch_indices))
            for i, true_label in enumerate(y_batch):
                outcome = 1 if predictions[i] == true_label else 0
                self._iv_records[true_label]['sizes'].append(current_train_size)
                self._iv_records[true_label]['outcomes'].append(outcome)
            train_indices.extend(batch_indices)
            current_index += batch_size

        # Initial Fit
        x_train = self.x_data[train_indices]
        y_train = self.y_data[train_indices]
        self.classifier.fit(x_train, y_train)

        while current_index < n_total:
            batch_indices = list(range(current_index, min(current_index + batch_size, n_total)))
            x_batch = self.x_data[batch_indices]
            y_batch = self.y_data[batch_indices]

            # Current training set size (used for computing accuracy function)
            current_train_size = len(train_indices)

            # Predict for the batch.
            predictions = self.classifier.predict(x_batch)

            # For each sample, record the prediction outcome in the IV records.
            for i, true_label in enumerate(y_batch):
                outcome = 1 if predictions[i] == true_label else 0
                self._iv_records[true_label]['sizes'].append(current_train_size)
                self._iv_records[true_label]['outcomes'].append(outcome)

            # Add the batch samples to the training set and retrain.
            train_indices.extend(batch_indices)
            x_train = self.x_data[train_indices]
            y_train = self.y_data[train_indices]
            self.classifier.fit(x_train, y_train)

            current_index += batch_size

    def compute_posterior(self, num_samples=1000, step_size=0.2, burn_in=10000, thin=50, random_seed=None, label=None, asymptote_flat_prior_borders=(0, 1), offset_factor_flat_prior_borders=(0, float('inf'))):
        """
        Computes the posterior distribution for the parameters (asymptote, offset_factor)
        based on the IV records via Markov Chain Monte Carlo (MCMC).

        For each label, given the recorded training set sizes and outcomes, the likelihood is:
            For a sample with training size s:
                p_correct = asymptote - offset_factor / s
            and its contribution:
                outcome * log(p_correct) + (1 - outcome) * log(1 - p_correct)
        Parameter constraints by default:
            - asymptote must lie in (0, 1).
            - offset_factor must be non-negative.
            - Also, p_correct must lie in (0, 1) for each sample, else the parameter set is rejected.

        The MCMC sampler (metropolis_hastings) from mcmc.py is used with the provided MCMC parameters.

        Parameters:
            num_samples : number of MCMC samples to return (after burn-in and thinning).
            step_size   : step size for the proposal distribution.
            burn_in     : number of initial samples to discard.
            thin        : interval for thinning the chain.
            random_seed : seed for reproducibility.
            label       : if specified, compute the posterior only for this label.
                          Otherwise, compute for all labels.
                          :param offset_factor_flat_prior_borders:
                          :param asymptote_flat_prior_borders:
        """
        assert num_samples != 0
        # NEW: Check if IV records have been computed. If not, warn and automatically call run_iv.
        total_records = sum(len(record['sizes']) for record in self._iv_records.values())
        if total_records == 0:
            batch_size = math.ceil(len(self.x_data) / 10)
            if len(self.x_data) >= 50:
                start_trainset_size = 1
            warnings.warn(
                f"compute_posterior was called before run_iv; no IV records exist yet. "
                f"Automatically running run_iv(start_trainset_size={start_trainset_size}, batch_size={batch_size})."
            )
            self.run_iv(start_trainset_size=start_trainset_size, batch_size=batch_size)

        def target_log_prob(theta, sizes, outcomes):
            asymptote, offset_factor = theta
            assert isinstance(asymptote_flat_prior_borders, tuple)
            assert isinstance(offset_factor_flat_prior_borders, tuple)
            asymptote_in_range = asymptote_flat_prior_borders[0] < asymptote < asymptote_flat_prior_borders[1]
            offset_factor_in_range = offset_factor_flat_prior_borders[0] < offset_factor < offset_factor_flat_prior_borders[1]
            if not asymptote_in_range or not offset_factor_in_range:
                return -np.inf
            log_prob = 0.0
            for s, outcome in zip(sizes, outcomes):
                p = asymptote - offset_factor / s
                if p >= 1:
                    return -np.inf
                elif p <= 0: # it might be a bit too heavy to set this -np.inf for small sizes
                    p = 1 / s # for very small sizes this is okay to happen, for high one not.
                    log_prob += np.log(p)  # TODO: This is an improper solution. Improve it.
                else:
                    log_prob += outcome * np.log(p) + (1 - outcome) * np.log(1 - p)
            return log_prob

        labels_to_process = [label] if label is not None else self._labels

        for lbl in labels_to_process:
            sizes = np.array(self._iv_records[lbl]['sizes'])
            outcomes = np.array(self._iv_records[lbl]['outcomes'])
            initial_value = np.array([0.91, 1.0])

            def target(theta):
                return target_log_prob(theta, sizes, outcomes)

            samples, acceptance_rate = metropolis_hastings(
                target_log_prob_fn=target,
                initial_value=initial_value,
                num_samples=num_samples,
                step_size=step_size,
                burn_in=burn_in,
                thin=thin,
                random_seed=random_seed
            )
            self._posterior[lbl] = (samples, acceptance_rate)
            # acc_samples = samples[:, 0]
            # off_samples = samples[:, 1]

        # TODO: Include priors //
        # TODO: Include time estimator xx
        # TODO: Issue warning when offset_factor is very high. Include instructions how to limit prior //
        # TODO: Issue warning when asymptote is lower than guessing probability. //

        # TODO: set these to their distributions //
        self.asymptotes = {label: self.get_label_dist(label, n=float('inf')) for label in self._labels}
        self.offset_factors = {label: self.get_offset_factor(label, recalc=True) for label in self._labels}

        for label in self._labels:
            if self.offset_factors[label].mean() >= 10:
                warnings.warn(
                    f"Offset factor for label {label} is very high. Consider limiting the prior."
                )
            if self.asymptotes[label].ppf(0.95) <= 1 / len(self._labels):
                warnings.warn(
                    f"Asymptote for label {label} is very low compared to guessing probability."
                    f"It classifies worse than a random guess."
                )

    def get_offset_factor(self, label, recalc=False):
        if not recalc:
            return self.offset_factors[label]
        samples, _ = self._posterior[label]
        num_bins = 50
        hist, bin_edges = np.histogram(samples[:, 1], bins=num_bins, density=True)
        dist = CustomHistogram((hist, bin_edges))
        return dist

    def get_label_dist(self, label, n):
        assert label in self._labels
        cache_key = (label, n)
        if cache_key in self._accuracy_cache:
            dist, raw_samples = self._accuracy_cache[cache_key]
        else:
            # NEW: Instead of raising an error if the posterior for the label is missing,
            # issue a warning and automatically compute it for that label.
            if label not in self._posterior:
                warnings.warn(
                    f"Posterior for label {label} not computed. Automatically computing posterior using default parameters."
                )
                self.compute_posterior(label=label)
            samples, _ = self._posterior[label]  # Samples of shape (num_samples, 2)
            if n == float('inf'):
                accuracy_samples = samples[:, 0]
            else:
                accuracy_samples = samples[:, 0] - samples[:, 1] / n
            accuracy_samples = np.clip(accuracy_samples, 0, 1)
            num_bins = 50
            hist, bin_edges = np.histogram(accuracy_samples, bins=num_bins, density=True)
            dist = CustomHistogram((hist, bin_edges))
            # dist = rv_histogram((hist, bin_edges))
            self._accuracy_cache[cache_key] = (dist, accuracy_samples)
            # raw_samples = accuracy_samples

        return dist

    def get_development(self, key, n=float('inf'), plot=False, confidence_range=0.95):
        self.development_confidence_range = confidence_range
        dist = self.get(key, n, plot, get_development=True)
        self.development_confidence_range = None  # without this it would behave mutable.
        return dist

    def get(
            self,
            key,
            n=float('inf'),
            plot=False,
            get_development=False
    ):
        if isinstance(key, str):
            if key.lower() == "bacc":
                key = [1/len(self._labels) for i in self._labels]
            elif key.lower() == "acc":
                key = [self._label_frequencies[i] for i in self._labels]
            else:
                raise ValueError(f'Unknown key. If key is string, key must be "acc" or "bacc"')
        pass
        # key can take any label, or a list of weights by which they are to be convoluted, or "acc" or "bacc".
        #   "acc" and "bacc" get then replaced with the weights.

        # always returns an instance of custom rv_histogram subclass.
        # from this any output can be gotten: mean/map/std/dist/samples. Also has a greater than function implemented.
        if get_development:
            means = []
            lower_bounds = []
            upper_bounds = []
            if n == float('inf'):
                n = 100
                warnings.warn(f'Getting development up to n={n}. To specify and silence this warning set n parameter.')
            for n_val in range(1, n):
                dist = self.get(key=key, n=n_val)  # get_development defaults to false which is desired
                mean_val = dist.mean()
                if self.development_confidence_range is not None:
                    confidence = self.development_confidence_range
                else:
                    print("Development confidence range not set. Using default 0.95%. "
                          "To set it and silence this message either set the development_confidence_range attribute or "
                          "use the get_development() method and pass the confidence_range parameter.")
                    confidence = 0.95  # default confidence level
                lb, ub = dist.interval(confidence=confidence) # TODO: This needs to be a parameter /x
                means.append(mean_val)
                lower_bounds.append(lb)
                upper_bounds.append(ub)
            self.plot(plot=plot, means=means, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
            return means, lower_bounds, upper_bounds


        if isinstance(key, list):
            assert len(key) == len(self._labels)
            dists = [self.get_label_dist(key, n) for key in self._labels]
            dist = weighted_sum_distribution(dists, weights=key)
            dist = CustomHistogram(dist._histogram)
            x = np.linspace(0, 1, 10000)
            self.plot(plot=plot, x=x, dist=dist.pdf(x))
            return dist
        elif key in self._labels:
            assert n != 0
            dist = self.get_label_dist(label=key, n=n)
            x = np.linspace(0, 1, 10000)
            self.plot(plot=plot, x=x, dist=dist.pdf(x))
            return dist
        else:
            raise ValueError(f'Unknown key. Key must be in labels ({self._labels}) or be "acc" or "bacc" or be list of weights')

    def plot(self, plot, x=None, **kwargs):
        if not plot:
            return
        if isinstance(plot, str):
            filename = plot  # TODO: clean the string xx
        else:
            filename = None

        import matplotlib.pyplot as plt

        for name, numbers in kwargs.items():
            if x is None:
                x = range(len(numbers))
            assert len(x) == len(numbers)
            plt.plot(x, numbers, label=name)  # Plot first list
        # plt.plot(x, y2, label='Dataset 2', color='red', marker='x')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()

        if filename:
            plt.savefig(filename)
        plt.show()
        plt.close()

    def get_bacc_dist(self, plot=False):
        return self.get(key='bacc', plot=plot)

    def get_acc_dist(self, plot=False):
        return self.get(key='acc', plot=plot)

    def get_label_accuracy(self, label, n=float('inf')):
        return self.get_label_dist(label, n)

