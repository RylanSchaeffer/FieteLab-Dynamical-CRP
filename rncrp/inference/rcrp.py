import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import tensorflow_probability as tfp
tfd = tfp.distributions
import torch
import torch.nn.functional
import torch.utils.data
from typing import Callable, Dict, Union

from rncrp.inference.base import BaseModel
from rncrp.helpers.dynamics import convert_dynamics_str_to_dynamics_obj
from rncrp.helpers.torch_helpers import assert_torch_no_nan_no_inf_is_real


class RecursiveCRP(BaseModel):
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
                 cutoff: float = 0.,
                 robbins_monro_cavi_updates: bool = True,
                 vi_param_initialization: str = 'observation',
                 which_prior_prob: str = 'DP',
                 update_new_cluster_parameters: bool = False,
                 **kwargs,
                 ):
        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.dynamics = convert_dynamics_str_to_dynamics_obj(
            dynamics_str='step',
            dynamics_params=dict(a=1., b=0.),
            implementation_mode='torch')
        self.component_prior_params = gen_model_params['component_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        self.model_str = model_str
        self.num_coord_ascent_steps_per_obs = num_coord_ascent_steps_per_obs
        self.numerically_optimize = numerically_optimize

        # Note: Learning rate is currently unused.
        if self.numerically_optimize:
            assert isinstance(learning_rate, float)
            assert learning_rate > 0.
        else:
            learning_rate = np.nan
        self.learning_rate = learning_rate
        self.cutoff = cutoff
        self.plot_dir = plot_dir
        assert vi_param_initialization in {'zero', 'observation'}
        self.vi_param_initialization = vi_param_initialization
        assert which_prior_prob in {'DP', 'variational'}
        self.which_prior_prob = which_prior_prob
        self.update_new_cluster_parameters = update_new_cluster_parameters
        self.robbins_monro_cavi_updates = robbins_monro_cavi_updates
        self.record_history = record_history
        self.fit_results = None

        # For some likelihoods e.g. von Mises-Fisher, we can compute (log)
        # probability of a new cluster since it doesn't depend on the
        # observation.
        self.log_prob_new_cluster = None

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

        cum_cluster_assignment_posteriors = torch.zeros(size=(max_num_clusters,),
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

            A_prefactor = self.gen_model_params['likelihood_params']['likelihood_cov_prefactor']

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
                    diag_covs=A_prefactor * torch.ones(2, max_num_clusters, obs_dim)))

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
                        fill_value=self.component_prior_params['beta_arg1'],
                        dtype=torch.float32),
                    arg2=torch.full(
                        size=(2, max_num_clusters, obs_dim),  # 2 for past & current
                        fill_value=self.component_prior_params['beta_arg2'],
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
                        fill_value=self.likelihood_params['likelihood_kappa'],
                        dtype=torch.float32,
                    )))

            # Recall, the log likelihood for a new cluster is:
            # log(C(k)*C(k)*C(0)) = 2 * log(C(k)) + log(C(0)).

            # Compute log(C(k)).
            normalizing_const_likelihood = self.compute_vonmisesfisher_normalization(
                dim=obs_dim,
                kappa=self.likelihood_params['likelihood_kappa'],
            )

            # Compute log(C(0)).
            normalizing_const_prior = self.compute_vonmisesfisher_normalization(
                dim=obs_dim,
                kappa=0.)

            # Compute log(C(k)*C(k)*C(0)).
            self.log_prob_new_cluster = 2. * np.log(normalizing_const_likelihood)\
                                        + np.log(normalizing_const_prior)

        else:
            raise NotImplementedError

        for obs_idx, torch_observation in enumerate(torch_observations):

            # Dataloader may return a dictionary. Select the observation from it.
            if isinstance(torch_observation, dict):
                torch_observation = torch_observation['observations'][0]  # Remove the batch index

            # print(f'Observation {obs_idx + 1}: ', torch_observation.numpy())

            # if obs_idx == 5:
            #     print()

            if obs_idx == 0:

                # First customer always goes at first table.
                cluster_assignment_priors[obs_idx, 0] = 1.
                variational_params['assignments']['probs'][obs_idx, 0] = 1.
                cluster_assignment_posterior = variational_params['assignments']['probs'][obs_idx, :].clone()
                num_clusters_posteriors[obs_idx, 0] = 1.

                self.dynamics.initialize_state(
                    customer_assignment_probs=cluster_assignment_posterior,
                    time=torch_observations_times[obs_idx])

                # Create parameters for each potential new cluster.
                initialize_cluster_params_fn(torch_observation=torch_observation,
                                             obs_idx=obs_idx,
                                             variational_params=variational_params)

                optimize_cluster_params_fn(
                    torch_observation=torch_observation,
                    obs_idx=obs_idx,
                    vi_idx=0,
                    variational_params=variational_params,
                    likelihood_params=self.gen_model_params['likelihood_params'],
                    cum_cluster_assignment_posteriors=cum_cluster_assignment_posteriors)

                # Overwrite old variational parameters with curr variational parameters.
                for variable, variable_variational_params_dict in variational_params.items():
                    if variable == 'assignments':
                        continue
                    for variational_param, variational_param_tensor in variable_variational_params_dict.items():
                        variational_param_tensor[0] = variational_param_tensor[1]

            else:

                # Step 1: Construct prior.
                # Step 1(i): Run dynamics.
                cluster_assignment_prior = self.dynamics.run_dynamics(
                    time_start=torch_observations_times[obs_idx - 1],
                    time_end=torch_observations_times[obs_idx])['N'].clone()

                # Step 1(ii): Add new table probability.
                cluster_assignment_prior[1:obs_idx + 1] += self.mixing_params['alpha'] * \
                                                           num_clusters_posteriors[obs_idx - 1, :obs_idx].clone()

                # Step 1(iii): Normalize.
                # print('Unnormalized cluster assignment prior: ', cluster_assignment_prior[:obs_idx + 1].numpy())
                cluster_assignment_prior /= torch.sum(cluster_assignment_prior)
                # print('Normalized cluster assignment prior: ', cluster_assignment_prior[:obs_idx + 1].numpy())
                assert_torch_no_nan_no_inf_is_real(cluster_assignment_prior)

                # print(cluster_assignment_prior.numpy()[:obs_idx + 1])

                # Step 1(iv): Sometimes, somehow, small negative numbers sneak in e.g. -2e-22
                # Identify them, test whether they're close to 0. If they are, replace with 0.
                # Otherwise, raise an assertion error.
                negative_indices = cluster_assignment_prior < 0.
                if torch.any(negative_indices):
                    # print(f'Smallest value: {torch.min(cluster_assignment_prior)}')
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
                variational_params['assignments']['probs'][obs_idx, :] = cluster_assignment_prior.clone()

                # Step 2(ii): Create parameter for potential new cluster.
                initialize_cluster_params_fn(torch_observation=torch_observation,
                                             obs_idx=obs_idx,
                                             variational_params=variational_params)

                # Step 3: Perform coordinate ascent on variational parameters.
                approx_lower_bounds = []
                for vi_idx in range(self.num_coord_ascent_steps_per_obs):

                    # print(f'Obs Idx: {obs_idx}, VI idx: {vi_idx}')

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

                            # time_2 = time.time()

                            optimize_cluster_params_fn(
                                torch_observation=torch_observation,
                                obs_idx=obs_idx,
                                vi_idx=vi_idx,
                                variational_params=variational_params,
                                likelihood_params=self.gen_model_params['likelihood_params'],
                                cum_cluster_assignment_posteriors=cum_cluster_assignment_posteriors)

                            # time_3 = time.time()
                            # print(f'Time2 - Time1: {time_2 - time_1}')
                            # print(f'Time3 - Time2: {time_3 - time_2}')

                    # print(torch_observations[:obs_idx + 1])
                    # print(variational_params['assignments']['probs'][obs_idx, :obs_idx+1])
                    # print(variational_params['means']['means'][1, obs_idx, :obs_idx+1])

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
                # print(f'Time4 - Time3: {time_4 - time_3}')

                # Step 5: Update dynamics state using new cluster assignment posterior.
                self.dynamics.update_state(
                    customer_assignment_probs=cluster_assignment_posterior,
                    time=torch_observations_times[obs_idx])

                time_5 = time.time()
                # print(f'Time5 - Time4: {time_5 - time_4}')

            cum_cluster_assignment_posteriors += cluster_assignment_posterior
            #
            # plt.close()
            # plt.scatter(1 + np.arange(obs_idx + 1),
            #             num_clusters_posteriors[obs_idx, :obs_idx + 1].numpy())
            # plt.ylabel('P(Num Clusters)')
            # plt.xlabel('Num Clusters')
            # plt.title(f'Obs Idx: {obs_idx + 1}')
            # plt.show()
            # plt.close()
            #
            # norm = plt.Normalize(0., obs_idx)
            # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            # fig.suptitle(f'Obs Idx: {obs_idx + 1}')
            # axes[0].scatter(variational_params['means']['means'][1, :obs_idx + 1, 0],
            #                 variational_params['means']['means'][1, :obs_idx + 1, 1],
            #                 # alpha=cluster_assignment_posterior[:obs_idx+1],
            #                 c=np.arange(obs_idx + 1),
            #                 norm=norm,
            #                 cmap='viridis',
            #                 # color='k',
            #                 # label='Centroids',
            #                 # marker='d',
            #                 # edgecolors='black',
            #                 s=20)
            # axes[0].set_xlim(-40, 40)
            # axes[0].set_ylim(-40, 40)
            # axes[0].set_title('Centroids')
            # axes[1].scatter(observations[:obs_idx + 1, 0],
            #                 observations[:obs_idx + 1, 1],
            #                 c=torch.argmax(variational_params['assignments']['probs'][np.newaxis, :obs_idx + 1, :],
            #                                dim=2).numpy()[0],
            #                 cmap='viridis',
            #                 norm=norm,
            #                 s=20)
            # # plt.legend()
            # axes[1].set_xlim(-40, 40)
            # axes[1].set_ylim(-40, 40)
            # axes[1].set_title('Observations')
            # plt.show()
            # plt.close()
            # print()

        # Add 1 because of indexing starts at 0.
        # TODO: Is this really the right way to determine the number of inferred clusters?
        num_inferred_clusters = 1 + torch.argmax(
            num_clusters_posteriors, dim=1).numpy()[-1].item()

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

    def initialize_cluster_params_multivariate_normal(self,
                                                      torch_observation: np.ndarray,
                                                      obs_idx: int,
                                                      variational_params: Dict[str, np.ndarray]):

        if self.vi_param_initialization == 'zero':
            variational_params['means']['means'].data[:, obs_idx, :] = 0.
        elif self.vi_param_initialization == 'observation':
            variational_params['means']['means'].data[:, obs_idx, :] = torch_observation
        else:
            raise ValueError
        assert_torch_no_nan_no_inf_is_real(variational_params['means']['means'])

    def initialize_cluster_params_product_bernoullis(self,
                                                     torch_observation: np.ndarray,
                                                     obs_idx: int,
                                                     variational_params: Dict[str, np.ndarray]):

        if self.vi_param_initialization == 'zero':
            variational_params['beta']['arg1'].data[:, obs_idx, :] = \
                self.component_prior_params['beta_arg1']
            variational_params['beta']['arg2'].data[:, obs_idx, :] = \
                self.component_prior_params['beta_arg2']

        elif self.vi_param_initialization == 'observation':
            variational_params['beta']['arg1'].data[:, obs_idx, :] = \
                self.component_prior_params['beta_arg1'] + torch_observation
            variational_params['beta']['arg2'].data[:, obs_idx, :] = \
                self.component_prior_params['beta_arg2'] + (1. - torch_observation)

        else:
            raise ValueError
        assert_torch_no_nan_no_inf_is_real(variational_params['beta']['arg1'])
        assert_torch_no_nan_no_inf_is_real(variational_params['beta']['arg2'])

    def initialize_cluster_params_vonmises_fisher(self,
                                                  torch_observation: np.ndarray,
                                                  obs_idx: int,
                                                  variational_params: Dict[str, np.ndarray]):

        # Data should already lie on sphere, so need to normalize.
        # if self.vi_param_initialization == 'zero':
        #     # For von-Mises, zero doesn't make any sense.
        #     variational_params['means']['means'].data[:, obs_idx, :] = torch_observation
        # elif self.vi_param_initialization == 'observation':
        #     variational_params['means']['means'].data[:, obs_idx, :] = torch_observation
        # else:
        #     raise ValueError
        variational_params['means']['means'].data[:, obs_idx, :] = torch_observation

        assert_torch_no_nan_no_inf_is_real(variational_params['means']['means'])

    @staticmethod
    def optimize_cluster_assignments_dirichlet_multinomial() -> None:
        raise NotImplementedError

    def optimize_cluster_assignments_multivariate_normal(self,
                                                         torch_observation: torch.Tensor,
                                                         obs_idx: int,
                                                         vi_idx: int,
                                                         cluster_assignment_prior: torch.Tensor,
                                                         variational_params: Dict[str, dict],
                                                         likelihood_params: Dict[str, float],
                                                         ) -> None:

        obs_dim = torch_observation.shape[0]
        sigma_obs_squared = self.likelihood_params['likelihood_cov_prefactor']

        # Term 1: log q(c_n = l | o_{<n})
        # Can get -inf here if probability of new cluster is 0.
        term_one = torch.log(cluster_assignment_prior[:obs_idx + 1])

        # Term 2: -o_n^T o_n / 2 sigma_obs^T
        term_two = -0.5 * torch.inner(
            torch_observation, torch_observation) / sigma_obs_squared
        assert_torch_no_nan_no_inf_is_real(term_two)

        # Term 3: mu_{nk}^T o_n / sigma_obs^2
        term_three = torch.einsum(
            'kd,d->k',
            variational_params['means']['means'][1, :obs_idx + 1, :],
            torch_observation) / sigma_obs_squared
        assert_torch_no_nan_no_inf_is_real(term_three)

        # Term 4:
        term_four = - 0.5 * torch.add(
            torch.sum(variational_params['means']['diag_covs'][1, :obs_idx + 1, :],
                      dim=1),  # Trace of diagonal covariance. Shape: (max number current clusters,)
            torch.einsum('ki,ki->k',
                         variational_params['means']['means'][1, :obs_idx + 1, :],
                         variational_params['means']['means'][1, :obs_idx + 1, :]),  # Tr[\mu_n \mu_n^T]
        ) / sigma_obs_squared
        assert_torch_no_nan_no_inf_is_real(term_four)

        # Term 4: -D * log(2 pi sigma_obs_squared) / 2
        term_five = -obs_dim * np.log(2 * np.pi * sigma_obs_squared) / 2.

        term_to_softmax = term_one + term_two + term_three + term_four + term_five

        # For the new cluster, the likelihood is N(0, likelihood cov + cluster mean prior cov)
        # Consequently, we need to overwrite the last index with the correct value.
        if self.which_prior_prob == 'DP':
            new_cluster_var = sigma_obs_squared + self.component_prior_params['centroids_prior_cov_prefactor']
            replacement_term_one = term_one[obs_idx]
            # Since mean mu_{nk} = 0, term two is 0 and we can skip.
            replacement_term_three = -0.5 * torch.square(torch.linalg.norm(torch_observation)) / \
                                     new_cluster_var
            replacement_term_four = -obs_dim * np.log(2 * np.pi * new_cluster_var) / 2.
            term_to_softmax[obs_idx] = replacement_term_one + replacement_term_three + replacement_term_four

        cluster_assignment_posterior = torch.nn.functional.softmax(
            term_to_softmax,  # shape: (max num clusters, )
            dim=0)

        # check that Bernoulli probs are all valid
        assert_torch_no_nan_no_inf_is_real(cluster_assignment_posterior)
        assert torch.all(0. <= cluster_assignment_posterior)
        assert torch.all(cluster_assignment_posterior <= 1.)

        variational_params['assignments']['probs'][obs_idx, :obs_idx + 1] = cluster_assignment_posterior

        # print('Cluster assignment posterior: ', np.round(cluster_assignment_posterior.numpy()[:obs_idx + 1], 2))

    def optimize_cluster_assignments_product_bernoullis(self,
                                                        torch_observation: torch.Tensor,
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

        # For the new cluster, the likelihood is N(0, likelihood cov + cluster mean prior cov)
        # Consequently, we need to overwrite the last index with the correct value.
        if self.which_prior_prob == 'DP':
            # new_cluster_var = sigma_obs_squared + self.component_prior_params['centroids_prior_cov_prefactor']
            # replacement_term_one = term_one[obs_idx]
            # # Since mean mu_{nk} = 0, term two is 0 and we can skip.
            # replacement_term_three = -0.5 * torch.square(torch.linalg.norm(torch_observation)) / \
            #                          new_cluster_var
            # replacement_term_four = -obs_dim * np.log(2 * np.pi * new_cluster_var) / 2.
            # term_to_softmax[obs_idx] = replacement_term_one + replacement_term_three + replacement_term_four
            raise NotImplementedError

        cluster_assignment_posterior_params = torch.nn.functional.softmax(
            term_to_softmax,  # shape: (max num clusters, )
            dim=0)

        # check that Bernoulli probs are all valid
        assert_torch_no_nan_no_inf_is_real(cluster_assignment_posterior_params)
        assert torch.all(0. <= cluster_assignment_posterior_params)
        assert torch.all(cluster_assignment_posterior_params <= 1.)

        # Shape: (curr max num clusters i.e. obs idx)
        variational_params['assignments']['probs'][obs_idx, :obs_idx + 1] = cluster_assignment_posterior_params

    def optimize_cluster_assignments_vonmises_fisher(self,
                                                     torch_observation: torch.Tensor,
                                                     obs_idx: int,
                                                     vi_idx: int,
                                                     cluster_assignment_prior: torch.Tensor,
                                                     variational_params: Dict[str, dict],
                                                     likelihood_params: Dict[str, float],
                                                     ) -> None:

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
        term_two = likelihood_params['likelihood_kappa'] * torch.einsum(
            'kd,d->k',
            torch_means,
            torch_observation)
        assert_torch_no_nan_no_inf_is_real(term_two)

        term_to_softmax = term_one + term_two

        # For the new cluster, the likelihood is N(0, likelihood cov + cluster mean prior cov)
        # Consequently, we need to overwrite the last index with the correct value.
        if self.which_prior_prob == 'DP':
            # Recall, we precompute the log probability of a new cluster in self.fit()
            # because, for the von-Mises-Fisher distribution, the probability of an
            # observation with a flat prior on the direction doesn't depend on the observation.
            term_to_softmax[obs_idx] = term_one[obs_idx] + self.log_prob_new_cluster

        cluster_assignment_posterior_params = torch.nn.functional.softmax(
            term_to_softmax,  # shape: (curr max num clusters i.e. obs idx, )
            dim=0)

        # check that Bernoulli probs are all valid
        assert_torch_no_nan_no_inf_is_real(cluster_assignment_posterior_params)
        assert torch.all(0. <= cluster_assignment_posterior_params)
        assert torch.all(cluster_assignment_posterior_params <= 1.)

        variational_params['assignments']['probs'][obs_idx, :obs_idx + 1] = cluster_assignment_posterior_params

    @staticmethod
    def optimize_cluster_params_dirichlet_multinomial(cum_cluster_assignment_posteriors: torch.Tensor,
                                                      ) -> None:
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

    def optimize_cluster_params_product_bernoullis(self,
                                                   torch_observation: torch.Tensor,
                                                   obs_idx: int,
                                                   vi_idx: int,
                                                   variational_params: Dict[str, dict],
                                                   likelihood_params: Dict[str, float],
                                                   cum_cluster_assignment_posteriors: torch.Tensor,
                                                   ) -> None:

        if self.update_new_cluster_parameters:
            max_cluster_idx_to_update = obs_idx + 1  # End index is exclusionary.
        else:
            # Recall, we only update the previous clusters' parameters.
            max_cluster_idx_to_update = obs_idx

        new_arg_1 = torch.add(
            variational_params['beta']['arg1'][0, :max_cluster_idx_to_update, :],  # previous parameter values
            torch.einsum(
                'c,o->co',
                variational_params['assignments']['probs'][obs_idx, :max_cluster_idx_to_update],  # Shape: (curr max num clusters ,)
                torch_observation,  # Shape: (obs dim,)
            )
        )
        assert_torch_no_nan_no_inf_is_real(new_arg_1)

        new_arg_2 = torch.add(
            variational_params['beta']['arg2'][0, :max_cluster_idx_to_update, :],  # previous parameter values
            torch.einsum(
                'c,o->co',
                variational_params['assignments']['probs'][obs_idx, :max_cluster_idx_to_update],  # Shape: (curr max num clusters,)
                1. - torch_observation,  # Shape: (obs dim,)
            ))
        assert_torch_no_nan_no_inf_is_real(new_arg_2)

        if not self.robbins_monro_cavi_updates:
            # Don't reduce effective step size.
            variational_params['beta']['arg1'][1, :obs_idx + 1, :] = new_arg_1
            variational_params['beta']['arg2'][1, :obs_idx + 1, :] = new_arg_2
        else:
            # Take linear combination: step size * new + (1-step size) * old
            step_size_per_cluster = self.compute_step_size(
                variational_params=variational_params,
                obs_idx=obs_idx,
                max_cluster_idx_to_update=max_cluster_idx_to_update,
                cum_cluster_assignment_posteriors=cum_cluster_assignment_posteriors,
            )

            scaled_new_arg_1 = torch.add(
                torch.multiply(
                    step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    new_arg_1,  # Shape (curr_max_cluster_idx, obs dim)
                ),
                torch.multiply(
                    1. - step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    variational_params['beta']['arg1'][0, :max_cluster_idx_to_update, :],
                )
            )

            scaled_new_arg_2 = torch.add(
                torch.multiply(
                    step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    new_arg_2,  # Shape (curr_max_cluster_idx, obs dim)
                ),
                torch.multiply(
                    1. - step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    variational_params['beta']['arg2'][0, :max_cluster_idx_to_update, :],
                )
            )

            # Shape: (max clusters to update, obs dim)
            variational_params['beta']['arg1'][1, :max_cluster_idx_to_update, :] = scaled_new_arg_1
            variational_params['beta']['arg2'][1, :max_cluster_idx_to_update, :] = scaled_new_arg_2

    def optimize_cluster_params_multivariate_normal(self,
                                                    torch_observation: torch.Tensor,
                                                    obs_idx: int,
                                                    vi_idx: int,
                                                    variational_params: Dict[str, dict],
                                                    likelihood_params: Dict[str, float],
                                                    cum_cluster_assignment_posteriors: torch.Tensor,
                                                    ) -> None:

        if self.update_new_cluster_parameters:
            max_cluster_idx_to_update = obs_idx + 1  # End index is exclusionary.
        else:
            # Recall, we only update the previous clusters' parameters.
            max_cluster_idx_to_update = obs_idx

        sigma_obs_squared = likelihood_params['likelihood_cov_prefactor']
        assert sigma_obs_squared > 0.

        prev_means_means = variational_params['means']['means'][0, :max_cluster_idx_to_update, :].clone()

        time_2_1 = time.time()
        prev_means_diag_precisions = 1. / variational_params['means']['diag_covs'][0, :max_cluster_idx_to_update, :]
        time_2_2 = time.time()
        # print(f'Time2.2 - Time2.1: {time_2_2 - time_2_1}')

        obs_dim = torch_observation.shape[0]
        curr_max_num_clusters = prev_means_means.shape[0]

        # Step 1: Compute updated covariances
        # Take I_{D \times D} and repeat to add a batch dimension
        # Shape (max clusters to update, obs_dim,)
        repeated_diag_eyes = torch.ones(size=(1, obs_dim)).repeat(
            curr_max_num_clusters, 1)
        weighted_diag_eyes = torch.multiply(
            variational_params['assignments']['probs'][obs_idx, :max_cluster_idx_to_update, np.newaxis],
            repeated_diag_eyes,
        ) / sigma_obs_squared
        new_mean_diag_precisions = torch.add(prev_means_diag_precisions, weighted_diag_eyes)
        new_means_diag_covs = 1. / new_mean_diag_precisions

        time_2_3 = time.time()
        # print(f'Time2.3 - Time2.2: {time_2_3 - time_2_2}')

        if not self.robbins_monro_cavi_updates:
            # Don't reduce effective step size.
            variational_params['means']['diag_covs'][1, :max_cluster_idx_to_update, :] = new_means_diag_covs
        else:
            step_size_per_cluster = self.compute_step_size(
                variational_params=variational_params,
                obs_idx=obs_idx,
                max_cluster_idx_to_update=max_cluster_idx_to_update,
                cum_cluster_assignment_posteriors=cum_cluster_assignment_posteriors,
            )

            # Shape: (curr max num clusters, obs dim)
            # Take linear combination: step size * new + (1-step size) * old
            scaled_new_means_diag_covs = torch.add(
                torch.multiply(
                    step_size_per_cluster[:, np.newaxis],  # Shape (curr max num clusters - 1, 1)
                    new_means_diag_covs,  # Shape (curr max num clusters - 1, obs dim)
                ),
                torch.multiply(
                    1. - step_size_per_cluster[:, np.newaxis],  # Shape (curr max num clusters, 1)
                    variational_params['means']['diag_covs'][0, :max_cluster_idx_to_update, :],
                )
            )
            # Shape: (curr max num obs - 1, obs dim)
            variational_params['means']['diag_covs'][1, :max_cluster_idx_to_update, :] = scaled_new_means_diag_covs

        # Slowest piece
        time_2_3_1 = time.time()
        # print(f'Time2.3.1 - Time2.3: {time_2_3_1 - time_2_3}')

        assert_torch_no_nan_no_inf_is_real(
            variational_params['means']['diag_covs'][1, :max_cluster_idx_to_update, :])

        time_2_4 = time.time()
        # print(f'Time2.4 - Time2.3: {time_2_4 - time_2_3}')

        # Step 2: Use updated covariances to compute updated means
        # Sigma_{n-1,l}^{-1} \mu_{n-1, l}
        term_one = torch.einsum(
            'ai,ai->ai',
            prev_means_diag_precisions,
            prev_means_means)

        # Need to add 1 when repeating because obs_idx starts at 0.
        term_two = torch.einsum(
            'k, kd->kd',
            variational_params['assignments']['probs'][obs_idx, :max_cluster_idx_to_update],
            # Shape: (max clusters to update, )
            torch_observation.reshape(1, obs_dim).repeat(max_cluster_idx_to_update, 1),
            # Shape: (max clusters to update, obs dim)
        ) / sigma_obs_squared

        new_means_means = torch.einsum(
            'bi, bi->bi',  # Technically, should be matrix multiplication, but we have diagonal matrix
            new_means_diag_covs,  # shape: (curr max num clusters -1, obs dim,)
            torch.add(term_one, term_two),  # shape: (curr max num clusters -1, obs dim)
        )

        time_2_5 = time.time()
        # print(f'Time2.5 - Time2.4: {time_2_5 - time_2_4}')
        assert_torch_no_nan_no_inf_is_real(new_means_means)

        if not self.robbins_monro_cavi_updates:
            # Don't reduce effective step size.
            variational_params['means']['means'][1, :max_cluster_idx_to_update, :] = new_means_means
        else:
            # Shape: (curr max num clusters, obs dim)
            # Take linear combination: step size * new + (1-step size) * old
            scaled_new_means_means = torch.add(
                torch.multiply(
                    step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    new_means_means,  # Shape (curr_max_cluster_idx, obs dim)
                ),
                torch.multiply(
                    1. - step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    variational_params['means']['means'][0, :max_cluster_idx_to_update, :],
                )
            )
            # Shape: (max clusters to update, obs dim)
            variational_params['means']['means'][1, :max_cluster_idx_to_update, :] = scaled_new_means_means

    def optimize_cluster_params_vonmises_fisher(self,
                                                torch_observation: torch.Tensor,
                                                obs_idx: int,
                                                vi_idx: int,
                                                variational_params: Dict[str, dict],
                                                likelihood_params: Dict[str, float],
                                                cum_cluster_assignment_posteriors: torch.Tensor,
                                                ) -> None:

        likelihood_kappa = likelihood_params['likelihood_kappa']

        if self.update_new_cluster_parameters:
            max_cluster_idx_to_update = obs_idx + 1  # End index is exclusionary.
        else:
            # Recall, we only update the previous clusters' parameters.
            max_cluster_idx_to_update = obs_idx

        rhs = torch.add(
            torch.multiply(variational_params['means']['concentrations'][0, :max_cluster_idx_to_update, :],
                           # (curr max num clusters - 1, 1)
                           variational_params['means']['means'][0, :max_cluster_idx_to_update, :],
                           # (curr max num clusters - 1, obs dim)
                           ),  # Shape: (curr max num clusters , obs dim)
            likelihood_kappa * torch.multiply(
                variational_params['assignments']['probs'][obs_idx, :max_cluster_idx_to_update, None],
                # Shape (curr max num clusters - 1, 1)
                torch_observation[None, :],  # Shape: (1, obs dim,)
            ),  # Shape: (curr max num clusters, obs dim)
        )  # Shape: (curr max num clusters, obs dim)

        magnitudes = torch.norm(rhs, dim=1, keepdim=True) # Shape: (curr max num clusters, 1)
        assert_torch_no_nan_no_inf_is_real(magnitudes)
        directions = rhs / magnitudes  # Shape: (max num clusters, obs dim)
        assert_torch_no_nan_no_inf_is_real(directions)

        if not self.robbins_monro_cavi_updates:
            # Don't reduce effective step size.
            variational_params['means']['concentrations'][1, :max_cluster_idx_to_update, :] = magnitudes
            variational_params['means']['means'][1, :max_cluster_idx_to_update, :] = directions
        else:
            step_size_per_cluster = self.compute_step_size(
                variational_params=variational_params,
                obs_idx=obs_idx,
                max_cluster_idx_to_update=max_cluster_idx_to_update,
                cum_cluster_assignment_posteriors=cum_cluster_assignment_posteriors,
            )

            scaled_directions = torch.add(
                torch.multiply(
                    step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    directions,  # Shape (curr_max_cluster_idx, 1)
                ),
                torch.multiply(
                    1. - step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    variational_params['means']['means'][0, :max_cluster_idx_to_update, :],
                )
            )

            scaled_magnitudes = torch.add(
                torch.multiply(
                    step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    magnitudes,  # Shape (curr_max_cluster_idx, obs dim)
                ),
                torch.multiply(
                    1. - step_size_per_cluster[:, np.newaxis],  # Shape (max clusters to update, 1)
                    variational_params['means']['concentrations'][0, :max_cluster_idx_to_update, :],
                )
            )

            variational_params['means']['concentrations'][1, :max_cluster_idx_to_update, :] = scaled_magnitudes
            variational_params['means']['means'][1, :max_cluster_idx_to_update, :] = scaled_directions

    @staticmethod
    def compute_step_size(variational_params: Dict[str, Dict[str, torch.Tensor]],
                          obs_idx: int,
                          max_cluster_idx_to_update: int,
                          cum_cluster_assignment_posteriors: torch.Tensor,
                          ) -> torch.Tensor:

        # Shape: (curr max num clusters - 1,)
        numerator = variational_params['assignments']['probs'][obs_idx, :max_cluster_idx_to_update]
        denominator = numerator + cum_cluster_assignment_posteriors[:max_cluster_idx_to_update]
        step_size_per_cluster = torch.divide(
            numerator,
            denominator)

        # If the cumulative probability mass is 0, the previous few lines
        # will divide 0/0 and result in NaN. Consequently, we mask those values.
        step_size_per_cluster[torch.isnan(step_size_per_cluster)] = 0.

        return step_size_per_cluster

    # def renormalize_after_cutoff(self):
    #
    #     # Renormalize
    #     if self.cutoff > 0:
    #         indices_to_zero = variational_params['assignments']['probs'][obs_idx,
    #                           :obs_idx + 1] < self.cutoff
    #         variational_params['assignments']['probs'][obs_idx, indices_to_zero] = 0.
    #         variational_params['assignments']['probs'][obs_idx, :obs_idx + 1] /= \
    #             torch.sum(variational_params['assignments']['probs'][obs_idx, :obs_idx + 1])
    #
    #         negative_indices = variational_params['assignments']['probs'][obs_idx,
    #                            :obs_idx + 1] < 0.
    #         if torch.any(negative_indices):
    #             variational_params['assignments']['probs'][obs_idx, :obs_idx + 1][
    #                 negative_indices] = 0.
    #             assert torch.all(cluster_assignment_prior >= 0.)


    @staticmethod
    def compute_vonmisesfisher_normalization(dim: int,
                                             kappa: float):
        if kappa > 0.:
            term1 = np.power(kappa, dim / 2 - 1)
            term2 = np.power(2 * np.pi, dim / 2)
            term3 = scipy.special.iv(
                dim / 2 - 1,  # order
                kappa,  # argument
            )
            normalizing_const = term1 / term2 / term3
        elif kappa == 0:
            # If kappa = 0., we need to compute surface area of sphere.
            # https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
            normalizing_const = 2. * np.power(np.pi, dim / 2.) / scipy.special.gamma( dim / 2.)
        else:
            raise ValueError(f'Impermissible kappa: {kappa}')
        return normalizing_const
