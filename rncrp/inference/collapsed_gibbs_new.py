import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import OneHotEncoder
from typing import Dict

from rncrp.inference.base import BaseModel


class CollapsedGibbsSampler2(BaseModel):

    """
    Collapsed Gibbs Sampling for Dirichlet Process Gaussian Mixture Model.

    Helpful resource: https://dp.tdhopper.com/collapsed-gibbs/
    """

    def __init__(self,
                 gen_model_params: Dict[str, Dict[str, float]],
                 model_str: str = 'CGS',
                 plot_dir: str = None,
                 num_passes: int = 100,
                 **kwargs,
                 ):
        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.component_prior_params = gen_model_params['component_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        self.num_passes = num_passes
        self.model_str = model_str
        self.plot_dir = plot_dir
        self.fit_results = None

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray):

        num_obs, obs_dim = observations.shape
        obs_idx_range = np.arange(num_obs)

        # TODO: Is there a better way to randomly initialize?
        cluster_assignment_posteriors = np.random.randint(
            low=0,
            high=2,
            size=(num_obs, num_obs)).astype(np.float32)
        cluster_assignment_posteriors = np.argmax(cluster_assignment_posteriors, axis=1)

        for pass_idx in range(self.num_passes):
            for obs_idx, observation in enumerate(observations):

                cluster_ids, num_obs_per_cluster = np.unique(
                    cluster_assignment_posteriors,
                    return_counts=True)

                # Exclude current observation from num obs per cluster.
                num_obs_per_cluster[cluster_ids == cluster_assignment_posteriors[obs_idx]] -= 1

                cluster_mean_per_cluster = np.zeros(shape=(len(cluster_ids) + 1, obs_dim))
                cluster_cov_per_cluster = np.zeros(shape=(len(cluster_ids) + 1, 1))

                for cluster_idx, (cluster_id, num_obs_in_cluster) in enumerate(zip(cluster_ids, num_obs_per_cluster)):
                    # Identify other points in cluster.
                    obs_in_cluster_id = observations[
                        (cluster_assignment_posteriors == cluster_id)
                        & (obs_idx_range != obs_idx)]

                    # Average other points in the cluster.
                    avg_obs_in_cluster_id = np.mean(obs_in_cluster_id, axis=0)

                    cluster_diag_prec = num_obs_in_cluster / self.likelihood_params['likelihood_cov_prefactor'] \
                                        + 1 / self.component_prior_params['centroids_prior_cov_prefactor']
                    cluster_diag_cov = 1. / cluster_diag_prec
                    cluster_cov_per_cluster[cluster_idx] = cluster_diag_cov

                    cluster_mean = cluster_diag_cov * num_obs_in_cluster * avg_obs_in_cluster_id \
                                   / self.likelihood_params['likelihood_cov_prefactor']
                    cluster_mean_per_cluster[cluster_idx] = cluster_mean

                cluster_cov_per_cluster[-1] = self.component_prior_params['centroids_prior_cov_prefactor'] \
                                              + self.likelihood_params['likelihood_cov_prefactor']

                log_likelihood_per_cluster = np.array([multivariate_normal.logpdf(
                        observation,
                        mean=cluster_mean_per_cluster[cluster_idx],
                        cov=cluster_cov_per_cluster[cluster_idx] * np.eye(obs_dim))
                    for cluster_idx in range(len(cluster_ids) + 1)])

                # If there are no other points in a cluster, then the log likelihood will be NaN
                # since the mean is NaN. Set these to negative infinity.
                log_likelihood_per_cluster[np.isnan(log_likelihood_per_cluster)] = -np.inf

                prior_per_cluster = np.concatenate([
                    num_obs_per_cluster,
                    np.array([self.mixing_params['alpha']])]).astype(np.float32)
                prior_per_cluster /= np.sum(prior_per_cluster)
                log_prior_per_cluster = np.log(prior_per_cluster)

                log_sampling_prob_per_cluster = log_likelihood_per_cluster + log_prior_per_cluster
                log_sampling_prob_per_cluster -= np.max(log_sampling_prob_per_cluster)
                sampling_prob_per_cluster = np.exp(log_sampling_prob_per_cluster)
                sampling_prob_per_cluster /= np.sum(sampling_prob_per_cluster)

                try:
                    assert np.alltrue(~np.isnan(sampling_prob_per_cluster))
                except AssertionError:
                    print()

                new_cluster_id = np.max(cluster_ids) + 1
                cluster_ids_plus_new = np.concatenate([
                    cluster_ids,
                    np.array([new_cluster_id])])
                new_cluster_assignment = np.random.choice(
                    cluster_ids_plus_new,
                    size=1,
                    p=sampling_prob_per_cluster,
                )

                cluster_assignment_posteriors[obs_idx] = new_cluster_assignment

            # Reset cluster ids.
            cluster_ids, num_obs_per_cluster = np.unique(
                cluster_assignment_posteriors,
                return_counts=True)
            # Find indices for size-biased sorting.
            sorted_indices_by_num_obs = np.argsort(num_obs_per_cluster)[::-1]
            # Replace size-sorted cluster id (e.g. 97, 83, 101, ...)
            # with integers starting at zero e.g. (0, 1, 2, ...).
            for cluster_idx, cluster_id in enumerate(cluster_ids[sorted_indices_by_num_obs]):
                cluster_assignment_posteriors[cluster_assignment_posteriors == cluster_id] = cluster_idx
            print(f'Pass: {pass_idx + 1}\tNum clusters: {len(cluster_ids)}\n'
                  f'Cluster sizes: {num_obs_per_cluster[sorted_indices_by_num_obs]}')

        # TODO Technically, we should recompute these, but I'm going to hope that the last
        # observation on the last pass doesn't change anything too much.
        params = dict(means=cluster_mean_per_cluster,
                      covs=cluster_cov_per_cluster * np.eye(obs_dim).reshape(
                          1, obs_dim, obs_dim).repeat(len(sampling_prob_per_cluster)))

        num_inferred_clusters = len(np.unique(cluster_assignment_posteriors))

        # Transform integers to one-hot.
        cluster_assignment_posteriors = OneHotEncoder(sparse=False).fit_transform(
            cluster_assignment_posteriors.reshape(-1, 1))

        # Reorder to "Left Ordered Form".
        shuffle_indices = np.lexsort(-cluster_assignment_posteriors[::-1])
        cluster_assignment_posteriors = cluster_assignment_posteriors[:, shuffle_indices]

        cluster_assignment_posteriors_running_sum = np.cumsum(
            cluster_assignment_posteriors,
            axis=0)

        self.fit_results = dict(
            cluster_assignment_posteriors=cluster_assignment_posteriors,
            cluster_assignment_posteriors_running_sum=cluster_assignment_posteriors_running_sum,
            num_inferred_clusters=num_inferred_clusters,
            parameters=params,
        )

        return self.fit_results

    def centroids_after_last_obs(self) -> np.ndarray:
        """
        Returns array of shape (num features, feature dimension)
        """
        return self.fit_results['parameters']['means']


class State(object):

    def __init__(self,
                 cluster_assignments: np.ndarray,  # Shape (num obs,)
                 ):
        self.cluster_assignments = cluster_assignments


class GaussianDistribution(object):

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def log_p(self, x):
        return multivariate_normal.logpdf(x=x, mean=self.mean, cov=self.cov)

    def sample(self):
        return multivariate_normal.rvs(mean=self.mean, cov=self.cov)
