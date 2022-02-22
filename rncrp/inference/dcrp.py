import time

# from mpmath import besselj
import numpy as np
import scipy
import sklearn.mixture
import tensorflow_probability as tfp

tfd = tfp.distributions
import torch
import torch.nn.functional
import torch.utils.data
from typing import Callable, Dict, Union

from rncrp.inference.base import BaseModel
from rncrp.helpers.dynamics import convert_dynamics_str_to_dynamics_obj
from rncrp.helpers.torch_helpers import assert_torch_no_nan_no_inf_is_real


class DynamicalCRP(BaseModel):
    """

    """

    def __init__(self,
                 gen_model_params: Dict[str, Dict],
                 model_str: str = 'Dynamical-CRP',
                 plot_dir: str = None,
                 num_coord_ascent_steps_per_obs: int = 3,
                 numerically_optimize: bool = False,
                 learning_rate: float = None,
                 record_history: bool = True,
                 **kwargs,
                 ):
        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.dynamics = convert_dynamics_str_to_dynamics_obj(
            dynamics_str=self.mixing_params['dynamics_str'],
            dynamics_params=self.mixing_params['dynamics_params'],
            implementation_mode='torch')
        self.component_prior_params = gen_model_params['component_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        self.model_str = model_str
        self.num_coord_ascent_steps_per_obs = num_coord_ascent_steps_per_obs
        self.numerically_optimize = numerically_optimize
        if self.numerically_optimize:
            assert isinstance(learning_rate, float)
            assert learning_rate > 0.
        else:
            learning_rate = np.nan
        self.learning_rate = learning_rate
        self.plot_dir = plot_dir
        self.record_history = record_history
        self.fit_results = None

    def fit(self,
            observations: Union[np.ndarray, torch.utils.data.DataLoader],
            observations_times: np.ndarray):

        if isinstance(observations, np.ndarray):
            num_obs, obs_dim = observations.shape
            torch_observations = torch.from_numpy(observations).float()
        elif isinstance(observations, torch.utils.data.DataLoader):
            num_obs = len(observations)
            obs_dim = observations.dataset[0]['observations'].shape[-1]
            torch_observations = observations
        else:
            raise ValueError

        max_num_clusters = num_obs

        torch_observations_times = torch.from_numpy(observations_times).float()

        cluster_assignment_priors = torch.zeros(size=(num_obs, max_num_clusters),
                                                dtype=torch.float32)

        num_clusters_posteriors = torch.zeros(size=(num_obs, max_num_clusters),
                                              dtype=torch.float32)

        if self.likelihood_params['distribution'] == 'dirichlet_multinomial':

            # initialize_cluster_params_fn = self.initialize_cluster_params_dirichlet_multinomial
            # optimize_cluster_assignments_fn = self.optimize_cluster_assignments_dirichlet_multinomial
            # optimize_cluster_params_fn = self.optimize_cluster_params_dirichlet_multinomial
            #
            # variational_params = dict(
            #     assignments=torch.full(
            #         size=(max_num_latents, obs_dim),
            #         fill_value=0.,
            #         dtype=torch.float32),
            #     concentrations=torch.full(size=(max_num_latents, obs_dim),
            #                               fill_value=np.nan,
            #                               dtype=torch.float32,
            #                               requires_grad=True))

            raise NotImplementedError

        elif self.likelihood_params['distribution'] == 'multivariate_normal':

            initialize_cluster_params_fn = self.initialize_cluster_params_multivariate_normal
            optimize_cluster_assignments_fn = self.optimize_cluster_assignments_multivariate_normal
            optimize_cluster_params_fn = self.optimize_cluster_params_multivariate_normal

            # TODO: What is the right covariance initialization? Isn't it just
            # A_prefactor = self.gen_model_params['component_prior_params']['centroids_prior_cov_prefactor'] \
            #               + self.gen_model_params['likelihood_params']['likelihood_cov_prefactor']
            A_prefactor = self.gen_model_params['likelihood_params']['likelihood_cov_prefactor']
            # Shape: (2 for old and new, max num clusters, obs dim)
            A_diag_covs = (A_prefactor * torch.ones(obs_dim).float()[None, None, :]).repeat(
                2, max_num_clusters, 1)

            variational_params = dict(
                assignments=dict(
                    probs=torch.full(
                        size=(num_obs, max_num_clusters),
                        fill_value=0.,
                        dtype=torch.float32)),
                means=dict(
                    means=torch.full(
                        size=(2, max_num_clusters, obs_dim),
                        fill_value=0.,
                        dtype=torch.float32),
                    diag_covs=A_diag_covs))

        elif self.likelihood_params['distribution'] == 'product_bernoullis':

            initialize_cluster_params_fn = self.initialize_cluster_params_product_bernoullis
            optimize_cluster_assignments_fn = self.optimize_cluster_assignments_product_bernoullis
            optimize_cluster_params_fn = self.optimize_cluster_params_product_bernoullis

            variational_params = dict(
                assignments=dict(
                    probs=torch.full(
                        size=(num_obs, max_num_clusters),
                        fill_value=0.,
                        dtype=torch.float32)),
                beta=dict(
                    arg1=torch.full(
                        size=(2, max_num_clusters, obs_dim),  # 2 for past & current
                        fill_value=0.,
                        dtype=torch.float32),
                    arg2=torch.full(
                        size=(2, max_num_clusters, obs_dim),  # 2 for past & current
                        fill_value=1.,
                        dtype=torch.float32,
                    )))

        elif self.likelihood_params['distribution'] == 'vonmises_fisher':

            initialize_cluster_params_fn = self.initialize_cluster_params_vonmises_fisher
            optimize_cluster_assignments_fn = self.optimize_cluster_assignments_vonmises_fisher
            optimize_cluster_params_fn = self.optimize_cluster_params_vonmises_fisher

            variational_params = dict(
                assignments=dict(
                    probs=torch.full(
                        size=(num_obs, max_num_clusters),
                        fill_value=0.,
                        dtype=torch.float32)),
                means=dict(
                    means=torch.full(
                        size=(2, max_num_clusters, obs_dim),
                        fill_value=0.,
                        dtype=torch.float32),
                    concentrations=torch.full(
                        size=(2, max_num_clusters, 1),
                        fill_value=1.,
                        dtype=torch.float32,
                    )))

        else:
            raise NotImplementedError

        for obs_idx, torch_observation in enumerate(torch_observations):

            # Dataloader may return a dictionary. Select the observation from it.
            if isinstance(torch_observation, dict):
                torch_observation = torch_observation['observations'][0]  # Remove the batch index

            if obs_idx == 0:

                # first customer has to go at first table
                cluster_assignment_priors[obs_idx, 0] = 1.
                variational_params['assignments']['probs'][obs_idx, 0] = 1.
                num_clusters_posteriors[obs_idx, 0] = 1.

                self.dynamics.initialize_state(
                    customer_assignment_probs=variational_params['assignments']['probs'][obs_idx, :],
                    time=torch_observations_times[obs_idx])

                # Create parameters for each potential new cluster
                initialize_cluster_params_fn(torch_observation=torch_observation,
                                             obs_idx=obs_idx,
                                             variational_params=variational_params)

            else:

                # Step 1: Construct prior.
                # Step 1(i): Run dynamics.
                cluster_assignment_prior = self.dynamics.run_dynamics(
                    time_start=torch_observations_times[obs_idx - 1],
                    time_end=torch_observations_times[obs_idx])['N']

                # Step 1(ii): Add new table probability.
                cluster_assignment_prior[1:obs_idx + 1] += self.mixing_params['alpha'] * \
                                                           num_clusters_posteriors[obs_idx - 1, :obs_idx].clone()

                # Step 1(iii): Normalize.
                cluster_assignment_prior /= torch.sum(cluster_assignment_prior)
                assert_torch_no_nan_no_inf_is_real(cluster_assignment_prior)

                # Step 1(iv): Sometimes, somehow, small negative numbers sneak in e.g. -2e-22
                # Identify them, test whether they're close to 0. If they are, replace with 0.
                # Otherwise, raise an assertion error.
                negative_indices = cluster_assignment_prior < 0.
                if torch.any(negative_indices):
                    print(f'Smallest value: {torch.min(cluster_assignment_prior)}')
                    # If the values are sufficiently close to 0, replace with 0.
                    # if torch.all(torch.isclose(cluster_assignment_prior[negative_indices],
                    #                            torch.tensor(0.),
                    #                            atol=1e-4)):
                    #     cluster_assignment_prior[negative_indices] = 0.
                    cluster_assignment_prior[negative_indices] = 0.
                    assert torch.all(cluster_assignment_prior >= 0.)

                # Record latent prior.
                assert_torch_no_nan_no_inf_is_real(cluster_assignment_prior)
                cluster_assignment_priors[obs_idx, :len(cluster_assignment_prior)] = cluster_assignment_prior

                # Step 2(i): Initialize assignments at prior.
                variational_params['assignments']['probs'][obs_idx, :] = cluster_assignment_prior

                # Step 2(ii): Create parameter for potential new cluster.
                initialize_cluster_params_fn(torch_observation=torch_observation,
                                             obs_idx=obs_idx,
                                             variational_params=variational_params)

                # Step 3: Perform coordinate ascent on variational parameters.
                approx_lower_bounds = []
                for vi_idx in range(self.num_coord_ascent_steps_per_obs):

                    print(f'Obs Idx: {obs_idx}, VI idx: {vi_idx}')

                    if self.numerically_optimize:
                        raise NotImplementedError
                    else:
                        with torch.no_grad():
                            time_1 = time.time()
                            optimize_cluster_assignments_fn(
                                torch_observation=torch_observation,
                                obs_idx=obs_idx,
                                vi_idx=vi_idx,
                                cluster_assignment_prior=cluster_assignment_prior,
                                variational_params=variational_params,
                                likelihood_params=self.gen_model_params['likelihood_params'])
                            time_2 = time.time()
                            optimize_cluster_params_fn(
                                torch_observation=torch_observation,
                                obs_idx=obs_idx,
                                vi_idx=vi_idx,
                                variational_params=variational_params,
                                likelihood_params=self.gen_model_params['likelihood_params'])
                            time_3 = time.time()
                            print(f'Time2 - Time1: {time_2 - time_1}')
                            print(f'Time3 - Time2: {time_3 - time_2}')

                # Overwrite old variational parameters with curr variational parameters
                for variable, variable_variational_params_dict in variational_params.items():
                    if variable == 'assignments':
                        continue
                    for variational_param, variational_param_tensor in variable_variational_params_dict.items():
                        variational_param_tensor[0] = variational_param_tensor[1]

                cluster_assignment_posterior = variational_params['assignments']['probs'][obs_idx, :].clone()

                # Step 4: Update posterior over number of clusters.
                # Use new approach with time complexity O(t).
                cum_table_assignment_posterior = torch.cumsum(
                    cluster_assignment_posterior[:obs_idx + 1],
                    dim=0)
                one_minus_cum_table_assignment_posterior = 1. - cum_table_assignment_posterior
                prev_num_clusters_posterior = num_clusters_posteriors[obs_idx - 1, :obs_idx]
                num_clusters_posteriors[obs_idx, :obs_idx] += torch.multiply(
                    cum_table_assignment_posterior[:-1],
                    prev_num_clusters_posterior)
                num_clusters_posteriors[obs_idx, 1:obs_idx + 1] += torch.multiply(
                    one_minus_cum_table_assignment_posterior[:-1],
                    prev_num_clusters_posterior)

                time_4 = time.time()
                print(f'Time4 - Time3: {time_4 - time_3}')

                # Step 5: Update dynamics state using new cluster assignment posterior.
                self.dynamics.update_state(
                    customer_assignment_probs=cluster_assignment_posterior,
                    time=torch_observations_times[obs_idx])

                time_5 = time.time()
                print(f'Time5 - Time4: {time_5 - time_4}')

        # Add 1 because of indexing starts at 0.
        num_inferred_clusters = 1 + torch.argmax(num_clusters_posteriors, dim=1).numpy()[-1].item()

        variational_parameters = {}
        for variable, variable_variational_params_dict in variational_params.items():
            if variable == 'assignments':
                continue
            for variational_param, variational_param_tensor in variable_variational_params_dict.items():
                variational_parameters[variational_param] = variational_param_tensor[1].numpy()

        self.fit_results = dict(
            cluster_assignment_priors=cluster_assignment_priors.numpy(),
            cluster_assignment_posteriors=variational_params['assignments']['probs'].detach().numpy(),
            num_clusters_posteriors=num_clusters_posteriors.numpy(),
            num_inferred_clusters=num_inferred_clusters,
            parameters=variational_parameters,
        )

        return self.fit_results

    def centroids_after_last_obs(self) -> np.ndarray:
        """
        Returns array of shape (num features, feature dimension)
        """
        return self.fit_results['parameters']['means']

    def initialize_cluster_params_dirichlet_multinomial(self,
                                                        torch_observation: torch.Tensor,
                                                        obs_idx: int,
                                                        variational_params: Dict[str, torch.Tensor],
                                                        epsilon: float = 10.):
        # assert_torch_no_nan_no_inf_is_real(torch_observation)
        # variational_params['concentrations'][0, obs_idx, :] = torch_observation + epsilon
        raise NotImplementedError

    @staticmethod
    def initialize_cluster_params_multivariate_normal(torch_observation: np.ndarray,
                                                      obs_idx: int,
                                                      variational_params: Dict[str, np.ndarray]):

        variational_params['means']['means'].data[:, obs_idx, :] = torch_observation
        assert_torch_no_nan_no_inf_is_real(variational_params['means']['means'])

    def initialize_cluster_params_product_bernoullis(self,
                                                     torch_observation: np.ndarray,
                                                     obs_idx: int,
                                                     variational_params: Dict[str, np.ndarray]):

        variational_params['beta']['arg1'].data[:, obs_idx, :] = \
            self.component_prior_params['beta_arg1'] + torch_observation
        assert_torch_no_nan_no_inf_is_real(variational_params['beta']['arg1'])
        variational_params['beta']['arg2'].data[:, obs_idx, :] = \
            self.component_prior_params['beta_arg2'] + (1. - torch_observation)
        assert_torch_no_nan_no_inf_is_real(variational_params['beta']['arg2'])

    @staticmethod
    def initialize_cluster_params_vonmises_fisher(torch_observation: np.ndarray,
                                                  obs_idx: int,
                                                  variational_params: Dict[str, np.ndarray]):

        # Data should already lie on sphere, so need to normalize.
        direction = torch_observation
        variational_params['means']['means'].data[:, obs_idx, :] = direction
        assert_torch_no_nan_no_inf_is_real(variational_params['means']['means'])

    @staticmethod
    def optimize_cluster_assignments_dirichlet_multinomial() -> None:
        raise NotImplementedError

    @staticmethod
    def optimize_cluster_assignments_multivariate_normal(torch_observation: torch.Tensor,
                                                         obs_idx: int,
                                                         vi_idx: int,
                                                         cluster_assignment_prior: torch.Tensor,
                                                         variational_params: Dict[str, dict],
                                                         likelihood_params: Dict[str, float],
                                                         ) -> None:

        sigma_obs_squared = likelihood_params['likelihood_cov_prefactor']

        # Term 1: log q(c_n = l | o_{<n})
        # Warning: can get -inf here if probability of new cluster is 0
        term_one = torch.log(cluster_assignment_prior[:obs_idx + 1])

        # Term 2: mu_{nk}^T o_n / sigma_obs^2
        term_two = torch.einsum(
            'kd,d->k',
            variational_params['means']['means'][1, :obs_idx + 1, :],
            torch_observation) / sigma_obs_squared
        assert_torch_no_nan_no_inf_is_real(term_two)

        # Term 3:
        term_three = - 0.5 * torch.add(
            torch.sum(variational_params['means']['diag_covs'][1, :obs_idx + 1, :],
                      dim=1),  # Trace of diagonal covariance. Shape: (num obs,)
            torch.einsum('ki,ki->k',
                         variational_params['means']['means'][1, :obs_idx + 1, :],
                         variational_params['means']['means'][1, :obs_idx + 1, :]),  # Tr[\mu_n \mu_n^T]
        ) / sigma_obs_squared
        assert_torch_no_nan_no_inf_is_real(term_three)

        term_to_softmax = term_one + term_two + term_three
        cluster_assignment_posterior_params = torch.nn.functional.softmax(
            term_to_softmax,  # shape: (max num clusters, )
            dim=0)

        # check that Bernoulli probs are all valid
        assert_torch_no_nan_no_inf_is_real(cluster_assignment_posterior_params)
        assert torch.all(0. <= cluster_assignment_posterior_params)
        assert torch.all(cluster_assignment_posterior_params <= 1.)

        variational_params['assignments']['probs'][obs_idx, :obs_idx + 1] = cluster_assignment_posterior_params

    @staticmethod
    def optimize_cluster_assignments_product_bernoullis(torch_observation: torch.Tensor,
                                                        obs_idx: int,
                                                        vi_idx: int,
                                                        cluster_assignment_prior: torch.Tensor,
                                                        variational_params: Dict[str, dict],
                                                        likelihood_params: Dict[str, float]):

        # Term 1: log q(c_n = l | o_{<n})
        # Warning: can get -inf here if probability of new cluster is 0
        term_one = torch.log(cluster_assignment_prior[:obs_idx + 1])

        # Shape: (curr max num clusters i.e. obs idx, obs dim)
        arg1_plus_arg2 = torch.add(
            variational_params['beta']['arg1'][1, :obs_idx + 1, :],
            variational_params['beta']['arg2'][1, :obs_idx + 1, :])
        digamma_arg1_plus_arg2 = torch.digamma(arg1_plus_arg2)
        digamma_arg1_minus_digamma_arg1_plus_arg2 = torch.sub(
            torch.digamma(variational_params['beta']['arg1'][1, :obs_idx + 1, :]),
            digamma_arg1_plus_arg2)
        digamma_arg2_minus_digamma_arg1_plus_arg2 = torch.sub(
            torch.digamma(variational_params['beta']['arg2'][1, :obs_idx + 1, :]),
            digamma_arg1_plus_arg2)

        # Term 2: \sum x_{nl}
        term_two_part_one = torch.einsum(
            'co,o->c',
            digamma_arg1_minus_digamma_arg1_plus_arg2,
            torch_observation,  # Shape: (obs dim,)
        )  # Shape: (curr max num clusters i.e. obs idx ,)
        term_two_part_two = torch.einsum(
            'co,o->c',
            digamma_arg2_minus_digamma_arg1_plus_arg2,
            1. - torch_observation,  # Shape: (obs dim, )
        )  # Shape: (curr max num clusters i.e. obs idx ,)

        # Shape: (curr max num clusters i.e. obs idx ,)
        term_two = term_two_part_one + term_two_part_two
        assert_torch_no_nan_no_inf_is_real(term_two)

        term_to_softmax = term_one + term_two
        cluster_assignment_posterior_params = torch.nn.functional.softmax(
            term_to_softmax,  # shape: (max num clusters, )
            dim=0)

        # check that Bernoulli probs are all valid
        assert_torch_no_nan_no_inf_is_real(cluster_assignment_posterior_params)
        assert torch.all(0. <= cluster_assignment_posterior_params)
        assert torch.all(cluster_assignment_posterior_params <= 1.)

        # Shape: (curr max num clusters i.e. obs idx)
        variational_params['assignments']['probs'][obs_idx, :obs_idx + 1] = cluster_assignment_posterior_params

    @staticmethod
    def optimize_cluster_assignments_vonmises_fisher(torch_observation: torch.Tensor,
                                                     obs_idx: int,
                                                     vi_idx: int,
                                                     cluster_assignment_prior: torch.Tensor,
                                                     variational_params: Dict[str, dict],
                                                     likelihood_params: Dict[str, float],
                                                     ) -> None:

        likelihood_kappa = likelihood_params['likelihood_kappa']

        # Term 1: log q(c_n = l | o_{<n})
        # Shape: (max num clusters, )
        # Warning: can get -inf here if probability of new cluster is 0
        term_one = torch.log(cluster_assignment_prior[:obs_idx + 1])

        # Shape: (max num clusters, obs dim)
        # TODO: Refactor to not use TensorFlow
        tf_means = tfd.VonMisesFisher(
            mean_direction=variational_params['means']['means'][1, :obs_idx + 1, :].numpy(),
            concentration=variational_params['means']['concentrations'][1, :obs_idx + 1, 0].numpy()).mean()
        torch_means = torch.from_numpy(tf_means.numpy())

        # Term 2: E[phi_{nk}]^T o_n / sigma_obs^2
        # Shape: (max num clusters, )
        term_two = likelihood_kappa * torch.einsum(
            'kd,d->k',
            torch_means,
            torch_observation)
        assert_torch_no_nan_no_inf_is_real(term_two)

        term_to_softmax = term_one + term_two
        cluster_assignment_posterior_params = torch.nn.functional.softmax(
            term_to_softmax,  # shape: (curr max num clusters i.e. obs idx, )
            dim=0)

        # check that Bernoulli probs are all valid
        assert_torch_no_nan_no_inf_is_real(cluster_assignment_posterior_params)
        assert torch.all(0. <= cluster_assignment_posterior_params)
        assert torch.all(cluster_assignment_posterior_params <= 1.)

        variational_params['assignments']['probs'][obs_idx, :obs_idx + 1] = cluster_assignment_posterior_params

    @staticmethod
    def optimize_cluster_params_dirichlet_multinomial() -> None:
        # def compute_likelihood_dirichlet_multinomial(self,
        #                                              torch_observation: torch.Tensor,
        #                                              obs_idx: int,
        #                                              variational_params: Dict[str, torch.Tensor], ):
        #     words_in_doc = torch.sum(torch_observation)
        #     total_concentrations_per_latent = torch.sum(
        #         variational_params['concentrations'][:obs_idx + 1], dim=1)
        #
        #     # Intermediate computations
        #     log_numerator = torch.log(words_in_doc) + log_beta(a=total_concentrations_per_latent, b=words_in_doc)
        #     log_beta_terms = log_beta(a=variational_params['concentrations'][:obs_idx + 1],
        #                               b=torch_observation)
        #
        #     log_x_times_beta_terms = torch.add(log_beta_terms, torch.log(torch_observation))
        #     log_x_times_beta_terms[torch.isnan(log_x_times_beta_terms)] = 0.
        #     log_denominator = torch.sum(log_x_times_beta_terms, dim=1)
        #
        #     assert_torch_no_nan_no_inf_is_real(log_denominator)
        #     log_likelihoods_per_latent = log_numerator - log_denominator
        #
        #     assert_torch_no_nan_no_inf_is_real(log_likelihoods_per_latent)
        #     likelihoods_per_latent = torch.exp(log_likelihoods_per_latent)
        #
        #     return likelihoods_per_latent, log_likelihoods_per_latent
        raise NotImplementedError

    @staticmethod
    def optimize_cluster_params_multivariate_normal(torch_observation: torch.Tensor,
                                                    obs_idx: int,
                                                    vi_idx: int,
                                                    variational_params: Dict[str, dict],
                                                    likelihood_params: Dict[str, float],
                                                    ) -> None:

        sigma_obs_squared = likelihood_params['likelihood_cov_prefactor']
        assert sigma_obs_squared > 0.

        prev_means_means = variational_params['means']['means'][0, :obs_idx + 1, :].clone()

        time_2_1 = time.time()
        prev_means_diag_precisions = 1. / variational_params['means']['diag_covs'][0, :obs_idx + 1, :]
        time_2_2 = time.time()
        print(f'Time2.2 - Time2.1: {time_2_2 - time_2_1}')

        obs_dim = torch_observation.shape[0]
        max_num_clusters = prev_means_means.shape[0]

        # Step 1: Compute updated covariances
        # Take I_{D \times D} and repeat to add a batch dimension
        # Shape (max num clusters, obs_dim,)
        repeated_diag_eyes = torch.ones(obs_dim).reshape(1, obs_dim).repeat(max_num_clusters, 1)
        weighted_diag_eyes = torch.multiply(
            variational_params['assignments']['probs'][obs_idx, :obs_idx + 1, None],  # shape (obs idx, 1)
            repeated_diag_eyes,  # shape (obs idx, 1)
        ) / sigma_obs_squared
        mean_diag_precisions = torch.add(prev_means_diag_precisions, weighted_diag_eyes)
        means_diag_covs = 1. / mean_diag_precisions

        time_2_3 = time.time()
        print(f'Time2.3 - Time2.2: {time_2_3 - time_2_2}')

        # No update on pytorch matrix square root
        # https://github.com/pytorch/pytorch/issues/9983#issuecomment-907530049
        # https://github.com/pytorch/pytorch/issues/25481
        # This matrix square root is the slowest part
        # new_means_stddevs = torch.stack([
        #     torch.from_numpy(scipy.linalg.sqrtm(gaussian_cov.detach().numpy()))
        #     for gaussian_cov in mean_covs])
        variational_params['means']['diag_covs'][1, :obs_idx + 1, :] = means_diag_covs

        # Slowest piece
        time_2_3_1 = time.time()
        print(f'Time2.3.1 - Time2.3: {time_2_3_1 - time_2_3}')

        assert_torch_no_nan_no_inf_is_real(
            variational_params['means']['diag_covs'][1, :, :])

        time_2_4 = time.time()
        print(f'Time2.4 - Time2.3: {time_2_4 - time_2_3}')

        # Step 2: Use updated covariances to compute updated means
        # Sigma_{n-1,l}^{-1} \mu_{n-1, l}
        term_one = torch.einsum(
            'ai,ai->ai',
            prev_means_diag_precisions,
            prev_means_means)

        # Need to add 1 when repeating because obs_idx starts at 0.
        term_two = torch.einsum(
            'k, kd->kd',
            variational_params['assignments']['probs'][obs_idx, :obs_idx + 1],  # Shape: (curr max num clusters, )
            torch_observation.reshape(1, obs_dim).repeat(obs_idx + 1, 1),  # Shape: (curr max num clusters, obs dim)
        ) / sigma_obs_squared

        new_means_means = torch.einsum(
            'bi, bi->bi',  # Technically, should be matrix multiplication, but we have diagonal matrix
            means_diag_covs,  # shape: (curr max num clusters, obs dim,)
            torch.add(term_one, term_two),  # shape: (curr max num clusters, obs dim)
        )

        time_2_5 = time.time()
        print(f'Time2.5 - Time2.4: {time_2_5 - time_2_4}')
        assert_torch_no_nan_no_inf_is_real(new_means_means)
        variational_params['means']['means'][1, :obs_idx + 1, :] = new_means_means

    @staticmethod
    def optimize_cluster_params_product_bernoullis(torch_observation: torch.Tensor,
                                                   obs_idx: int,
                                                   vi_idx: int,
                                                   variational_params: Dict[str, dict],
                                                   likelihood_params: Dict[str, float],
                                                   ) -> None:

        variational_params['beta']['arg1'][1, :, :] = torch.add(
            variational_params['beta']['arg1'][0, :, :],  # previous parameter values
            torch.einsum(
                'c,o->co',
                variational_params['assignments']['probs'][obs_idx],  # Shape: (max num clusters,)
                torch_observation,  # Shape: (obs dim,)
            )
        )
        assert_torch_no_nan_no_inf_is_real(variational_params['beta']['arg1'][1, :, :])

        variational_params['beta']['arg2'][1, :, :] = torch.add(
            variational_params['beta']['arg2'][0, :, :],  # previous parameter values
            torch.einsum(
                'c,o->co',
                variational_params['assignments']['probs'][obs_idx],  # Shape: (max num clusters,)
                1. - torch_observation,  # Shape: (obs dim,)
            ))
        assert_torch_no_nan_no_inf_is_real(variational_params['beta']['arg2'][1, :, :])

    @staticmethod
    def optimize_cluster_params_vonmises_fisher(torch_observation: torch.Tensor,
                                                obs_idx: int,
                                                vi_idx: int,
                                                variational_params: Dict[str, dict],
                                                likelihood_params: Dict[str, float],
                                                ) -> None:

        likelihood_kappa = likelihood_params['likelihood_kappa']

        rhs = torch.add(
            torch.multiply(variational_params['means']['concentrations'][0, :obs_idx + 1, :],  # (max num clusters, 1)
                           variational_params['means']['means'][0, :obs_idx + 1, :],  # (max num clusters, obs dim)
                           ),  # Shape: (max num clusters, obs dim)
            likelihood_kappa * torch.multiply(
                variational_params['assignments']['probs'][obs_idx, :obs_idx + 1, None],  # Shape (max num clusters, 1)
                torch_observation[None, :],  # Shape: (1, obs dim,)
            ),  # Shape: (max num clusters, obs dim)
        )  # Shape: (max num clusters, obs dim)

        magnitudes = torch.norm(rhs, dim=1)  # Shape: (max num clusters,)
        assert_torch_no_nan_no_inf_is_real(magnitudes)
        directions = rhs / magnitudes[:, None]  # Shape: (max num clusters, obs dim)
        assert_torch_no_nan_no_inf_is_real(directions)
        variational_params['means']['concentrations'][1, :obs_idx + 1, 0] = magnitudes
        variational_params['means']['means'][1, :obs_idx + 1, :] = directions
