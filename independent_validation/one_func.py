from .iv_file import IV
import math

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
        asymptote_prior=(0, 1),
        offset_factor_prior=(0, float('inf')),
):
    assert output.lower() in ['dist', 'distribution', 'map', 'mean', 'std']
    iv_1 = IV(x_data=X, y_data=y, classifier=classifier)
    if iv_batch_size is None and iv_n_batches is not None:
        iv_batch_size = math.ceil(len(X) / iv_n_batches)
    if iv_batch_size is None and iv_n_batches is None:
        iv_batch_size = 1
    iv_1.run_iv(start_trainset_size=iv_start_trainset_size, batch_size=iv_batch_size)
    iv_1.compute_posterior(num_samples=mcmc_num_samples, step_size=mcmc_step_size, burn_in=mcmc_burn_in, thin=mcmc_thin,
                           random_seed=mcmc_random_seed, asymptote_flat_prior_borders=asymptote_prior, offset_factor_flat_prior_borders=offset_factor_prior)
    dist = iv_1.get(key=key, n=n, plot=plot)

    if output.lower() == 'dist' or output.lower() == 'distribution':
        return dist
    elif output.lower() == 'map':
        return dist.map()
    elif output.lower() =='mean':
        return dist.mean()
    else:
        return dist.std()