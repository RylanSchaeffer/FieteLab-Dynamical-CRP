import numpy as np
import scipy.stats
from scipy.stats import multivariate_normal
from sklearn.preprocessing import OneHotEncoder
from typing import Dict

from rncrp.inference.base import BaseModel


class State(object):

    def __init__(self,
                 num_obs: int,
                 # gen_model_params: Dict[str, Dict[str, float]],
                 max_num_clusters: int = None,
                 ):

        if max_num_clusters is None:
            max_num_clusters = num_obs

        # self.gen_model_params = gen_model_params
        # self.sigma_obs_squared = gen_model_params['likelihood_params'][
        #     'likelihood_cov_prefactor']
        # self.sigma_mean_squared = gen_model_params['component_prior_params'][
        #     'centroids_prior_cov_prefactor']

        # TODO: Is there a better way to initialize?
        cluster_assignments_one_hot = np.random.randint(
            low=0,
            high=2,
            size=(num_obs, max_num_clusters)).astype(np.float32)
        self.cluster_assignments = np.argmax(cluster_assignments_one_hot, axis=1)

    def rename_cluster_ids(self):
        """
        This function's purpose is to replace the cluster IDs e.g. 97, 83, 107, ...
        with integers starting at zero e.g. 0, 1, 2, ...
        """

        # Recompute cluster ids.
        cluster_ids, num_obs_per_cluster = np.unique(
            self.cluster_assignments,
            return_counts=True)

        # Find indices for size-biased sorting, largest to smallest.
        sorted_indices_by_num_obs = np.argsort(num_obs_per_cluster)[::-1]
        # Replace size-sorted cluster id (e.g. 97, 83, 101, ...)
        # with integers starting at zero e.g. (0, 1, 2, ...).
        new_cluster_assignments = np.full_like(
            self.cluster_assignments,
            fill_value=-1.)
        for cluster_idx, cluster_id in enumerate(cluster_ids[sorted_indices_by_num_obs]):
            new_cluster_assignments[self.cluster_assignments == cluster_id] = cluster_idx
        self.cluster_assignments = new_cluster_assignments.copy()


class CollapsedGibbsSamplerNew(BaseModel):
    """
    Collapsed Gibbs Sampling for Dirichlet Process-Gaussian Mixture Model.

    Helpful resources:
        https://dp.tdhopper.com/collapsed-gibbs/
        http://gregorygundersen.com/blog/2020/11/18/bayesian-mvn/
    """

    def __init__(self,
                 gen_model_params: Dict[str, Dict[str, float]],
                 model_str: str = 'CGS',
                 plot_dir: str = None,
                 num_samples: int = 23,
                 burn_in_steps: int = 20000,
                 thinning_num_steps: int = 1000,
                 **kwargs,
                 ):
        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.component_prior_params = gen_model_params['component_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']

        self.num_samples = num_samples
        self.num_burn_in_steps = burn_in_steps
        self.num_thinning_steps = thinning_num_steps
        self.total_steps = self.num_burn_in_steps + self.num_samples * self.num_thinning_steps

        self.obs_idx_range = None

        self.model_str = model_str
        self.plot_dir = plot_dir
        self.fit_results = None

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray):

        num_obs, obs_dim = observations.shape

        # Preallocate for speed.
        self.obs_idx_range = np.arange(num_obs)

        state = State(num_obs=num_obs)

        cluster_assignments_mcmc_samples = np.zeros(shape=(self.num_samples, num_obs))
        sample_idx = 0

        for step_idx in range(self.total_steps):

            if step_idx % 100 == 0:
                print(f'Monte Carlo State Idx: {step_idx}')

            # Modifies state in-place.
            self.gibbs_step(observations=observations,
                            state=state)

            if step_idx < self.num_burn_in_steps:
                continue

            if (step_idx - self.num_burn_in_steps) % self.num_thinning_steps == 0:
                cluster_assignments_mcmc_samples[sample_idx] = state.cluster_assignments.copy()
                sample_idx += 1
                if sample_idx == cluster_assignments_mcmc_samples.shape[0]:
                    break

        num_inferred_clusters_mcmc_samples = np.array([
            len(np.unique(cluster_assignments_sample))
            for cluster_assignments_sample in cluster_assignments_mcmc_samples])
        mean_num_inferred_clusters = np.mean(num_inferred_clusters_mcmc_samples)

        # Transform integers to one-hot.
        cluster_assignments_one_hot_mcmc_samples = []
        params_mcmc_samples = dict(means=[], covs=[])
        for sample_idx in range(self.num_samples):

            cluster_assignments_sample = cluster_assignments_mcmc_samples[sample_idx]

            cluster_assignments_one_hot_sample = OneHotEncoder(sparse=False).fit_transform(
                cluster_assignments_sample.reshape(-1, 1))

            # Reorder to "Left Ordered Form".
            shuffle_indices = np.lexsort(-cluster_assignments_one_hot_sample[::-1])
            cluster_assignments_one_hot_sample = cluster_assignments_one_hot_sample[:, shuffle_indices]

            cluster_assignments_one_hot_mcmc_samples.append(
                cluster_assignments_one_hot_sample)

            # Cluster means
            for cluster_id in np.unique(cluster_assignments_sample):
                indices_in_cluster = cluster_assignments_sample == cluster_id
                assert np.sum(indices_in_cluster) > 0
                cluster_mean = np.mean(observations[indices_in_cluster], axis=0)
                params_mcmc_samples['means'].append(cluster_mean)
                # TODO: implement cluster covariances
                cluster_cov = np.cov(observations[indices_in_cluster].T)
                params_mcmc_samples['covs'].append(cluster_cov)

        self.fit_results = dict(
            cluster_assignments_mcmc_samples=cluster_assignments_mcmc_samples,
            cluster_assignments_one_hot_mcmc_samples=cluster_assignments_one_hot_mcmc_samples,
            cluster_assignment_posteriors_running_sum=None,
            num_inferred_clusters_mcmc_samples=num_inferred_clusters_mcmc_samples,
            parameters=params_mcmc_samples,
        )

        return self.fit_results

    def centroids_after_last_obs(self) -> np.ndarray:
        """
        Returns array of shape (num features, feature dimension)
        """
        return self.fit_results['parameters']['means']

    def compute_cond_assignment(self,
                                observations: np.ndarray,
                                obs_idx: np.ndarray,
                                state: State):

        num_obs, obs_dim = observations.shape

        # Ensure the cluster IDs have nice low integer values.
        state.rename_cluster_ids()

        cluster_ids, num_obs_per_cluster = np.unique(
            state.cluster_assignments,
            return_counts=True)

        # Exclude current observation from number of observations per cluster.
        num_obs_per_cluster[cluster_ids == state.cluster_assignments[obs_idx]] -= 1

        # Convert from ints (counts) to float to be able to normalize.
        prior_per_cluster = np.append(
            num_obs_per_cluster,
            self.mixing_params['alpha']).astype(np.float32)
        prior_per_cluster /= np.sum(prior_per_cluster)
        log_prior_per_cluster = np.log(prior_per_cluster)

        # Add 1 for consideration of new cluster.
        cluster_mean_per_cluster = np.zeros(shape=(len(cluster_ids) + 1, observations.shape[1]))
        cluster_cov_per_cluster = np.zeros(shape=(len(cluster_ids) + 1, 1))

        for cluster_idx, (cluster_id, num_obs_in_cluster) in enumerate(zip(cluster_ids, num_obs_per_cluster)):

            # Identify other points in cluster.
            other_obs_in_cluster = observations[
                (state.cluster_assignments == cluster_id)
                & (self.obs_idx_range != obs_idx)]

            # Average other points in the cluster.
            # This can produce NaN if there are no other observations in this cluster.
            avg_obs_in_cluster = np.mean(other_obs_in_cluster, axis=0)

            cluster_diag_prec = (num_obs_in_cluster / self.likelihood_params['likelihood_cov_prefactor']) \
                                + (1. / self.component_prior_params['centroids_prior_cov_prefactor'])
            cluster_diag_cov = 1. / cluster_diag_prec
            cluster_cov_per_cluster[cluster_idx] = cluster_diag_cov

            cluster_mean = cluster_diag_cov * num_obs_in_cluster * avg_obs_in_cluster \
                           / self.likelihood_params['likelihood_cov_prefactor']
            cluster_mean_per_cluster[cluster_idx] = cluster_mean

        # Set mean and cov for the possible new cluster.
        # Note: Don't need to set mean because the prior mean is 0.
        cluster_cov_per_cluster[-1] = self.component_prior_params['centroids_prior_cov_prefactor'] \
                                      + self.likelihood_params['likelihood_cov_prefactor']

        # import matplotlib.pyplot as plt
        #
        # plt.close()
        # plt.scatter(observations[obs_idx, 0],
        #             observations[obs_idx, 1],
        #             label='Observation')
        # plt.scatter(cluster_mean_per_cluster[:, 0],
        #             cluster_mean_per_cluster[:, 1],
        #             c=np.arange(len(cluster_ids)+1),
        #             label='Clusters')
        # plt.legend()
        # plt.show()

        log_likelihood_per_cluster = np.array([multivariate_normal.logpdf(
            observations[obs_idx],
            mean=cluster_mean_per_cluster[cluster_idx],
            cov=cluster_cov_per_cluster[cluster_idx] * np.eye(observations.shape[1]))
            for cluster_idx in range(len(cluster_ids) + 1)])

        # If there are no other points in a cluster, then the log likelihood will be NaN
        # since the mean is NaN. Set these to negative infinity.
        log_likelihood_per_cluster[np.isnan(log_likelihood_per_cluster)] = -np.inf

        # For the new cluster, use likelihood of N(0, (sigma_obs_sqrd + sigma_mean_sqrd) * Eye)
        new_cluster_var = self.likelihood_params['likelihood_cov_prefactor']\
                          + self.component_prior_params['centroids_prior_cov_prefactor']
        log_likelihood_per_cluster[-1] = multivariate_normal.logpdf(
            observations[obs_idx],
            mean=np.zeros(obs_dim),
            cov=new_cluster_var * np.eye(obs_dim))

        log_sampling_prob_per_cluster = log_likelihood_per_cluster + log_prior_per_cluster

        # For numerical stability, first subtract max.
        log_sampling_prob_per_cluster -= np.max(log_sampling_prob_per_cluster)

        # Compute softmax.
        sampling_prob_per_cluster = np.exp(log_sampling_prob_per_cluster)
        sampling_prob_per_cluster /= np.sum(sampling_prob_per_cluster)

        try:
            assert np.all(~np.isnan(sampling_prob_per_cluster))
        except AssertionError:
            print()

        return sampling_prob_per_cluster

    def gibbs_step(self,
                   observations: np.ndarray,
                   state: State):

        # Choose which observation to update.
        obs_idx = np.random.choice(observations.shape[0])

        cluster_assignment_distribution = self.compute_cond_assignment(
            observations=observations,
            obs_idx=obs_idx,
            state=state,
        )

        state.cluster_assignments[obs_idx] = np.random.choice(
            len(cluster_assignment_distribution),
            size=1,
            p=cluster_assignment_distribution)

    def geweke_test_wrong(self,
                          cluster_assignments_mcmc_samples: np.ndarray,
                          observations: np.ndarray,
                          num_samples: int = None):

        if num_samples is None:
            num_samples = cluster_assignments_mcmc_samples.shape[0]

        num_obs, obs_dim = observations.shape
        state = State(num_obs=num_obs)

        forward_statistics = []
        gibbs_forward_statistics = []

        for sample_idx in range(num_samples):

            # Choose sampled assignments from MCMC.
            # sample_idx = np.random.choice(
            #     cluster_assignments_samples.shape[0])
            # sample_idx =
            sampled_cluster_assignments = cluster_assignments_mcmc_samples[sample_idx, :]
            state.cluster_assignments[:] = sampled_cluster_assignments.copy()

            # Compute statistics from one MCMC cluster assignments
            # First sample cluster means.
            num_clusters_in_sample = len(np.unique(sampled_cluster_assignments))
            forward_statistics.append(num_clusters_in_sample)

            # Take several Gibbs steps to move from MCMC cluster assignments
            # to new MCMC cluster assignments.
            for _ in range(self.num_thinning_steps):
                self.gibbs_step(state=state, observations=observations)
            gibbs_forward_cluster_assignments = state.cluster_assignments.copy()

            # Compute statistics from new MCMC cluster assignments
            num_clusters_in_forward_sample = len(np.unique(gibbs_forward_cluster_assignments))
            gibbs_forward_statistics.append(num_clusters_in_forward_sample)

        import matplotlib.pyplot as plt

        plt.plot(np.sort(forward_statistics), np.sort(gibbs_forward_statistics))
        min_value = min(np.min(forward_statistics), np.min(gibbs_forward_statistics))
        max_value = max(np.max(forward_statistics), np.max(gibbs_forward_statistics))
        plt.title('Num Inferred Clusters')
        plt.xlabel('Forward: Num Inferred Clusters')
        plt.ylabel('Gibbs Forward: Num Inferred Clusters')
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
        plt.show()
        plt.close()
        print(10)

    def geweke_test(self,
                    num_obs: int,
                    obs_dim: int,
                    num_repeats: int = 25, ):

        forward_statistics = []
        gibbs_statistics = []
        from rncrp.data.synthetic import sample_mixture_model
        for repeat_idx in range(num_repeats):

            print(f'Gweke Test Repeat Idx: {repeat_idx}')

            # Forward sample.
            mixture_model_results = sample_mixture_model(
                num_obs=num_obs,
                obs_dim=obs_dim,
                mixing_prior_str='rncrp',
                mixing_distribution_params={'alpha': self.mixing_params['alpha'],
                                            'beta': self.mixing_params['beta'],
                                            'dynamics_str': 'step'},
                component_prior_str='gaussian',
                component_prior_params={
                    'centroids_prior_cov_prefactor': self.component_prior_params['centroids_prior_cov_prefactor'],
                    'likelihood_cov_prefactor': self.likelihood_params['likelihood_cov_prefactor']}
            )

            # Compute statistics from forward sample.
            num_clusters_in_sample = len(np.unique(mixture_model_results['cluster_assignments']))
            forward_statistics.append(num_clusters_in_sample)

            fit_results = self.fit(
                observations=mixture_model_results['observations'],
                observations_times=None)

            # Compute statistics from new MCMC cluster assignments.
            # Take the median.
            num_clusters_in_forward_sample = np.median(
                fit_results['num_inferred_clusters_mcmc_samples'])
            gibbs_statistics.append(num_clusters_in_forward_sample)

        gweke_test_results = dict(
            gibbs_statistics=gibbs_statistics,
            forward_statistics=forward_statistics,
        )

        return gweke_test_results
