import numpy as np
import sklearn.mixture
from typing import Callable, Dict

import torch

from rncrp.inference.base import BaseModel
from rncrp.helpers.dynamics import convert_dynamics_str_to_dynamics_obj
from rncrp.helpers.numpy_helpers import assert_np_no_nan_no_inf_is_real


class RecursiveNonstationaryCRP(BaseModel):
    """
    Variational Inference for Dirichlet Process Gaussian Mixture Model, as
    proposed by Blei and Jordan (2006).

    Wrapper around scikit-learn's implementation.
    """

    def __init__(self,
                 gen_model_params: Dict[str, Dict],
                 model_str: str = 'RN-CRP',
                 plot_dir: str = None,
                 num_coord_ascent_steps_per_obs: int = 3,
                 numerically_optimize: bool = False,
                 learning_rate: float = None,
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
        self.feature_prior_params = gen_model_params['feature_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        self.model_str = model_str
        self.plot_dir = plot_dir
        self.fit_results = None
        self.num_coord_ascent_steps_per_obs = num_coord_ascent_steps_per_obs
        self.numerically_optimize = numerically_optimize
        if self.numerically_optimize:
            assert isinstance(learning_rate, float)
            assert learning_rate > 0.
        else:
            learning_rate = np.nan
        self.learning_rate = learning_rate

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray):

        num_obs, obs_dim = observations.shape
        max_num_latents = num_obs

        torch_observations = torch.from_numpy(observations)
        torch_observations_times = torch.from_numpy(observations_times)

        cluster_assignment_priors = torch.zeros((num_obs, max_num_latents),
                                                dtype=np.float32)
        cluster_assignment_priors[0, 0] = 1.

        cluster_assignment_posteriors = torch.zeros((num_obs, max_num_latents),
                                                    dtype=np.float32, )

        cluster_assignment_posteriors_running_sum = torch.zeros((num_obs, max_num_latents),
                                                                dtype=np.float32)

        num_cluster_posteriors = torch.zeros((num_obs, max_num_latents),
                                             dtype=np.float32)

        if self.likelihood_params['distribution'] == 'multivariate_normal':
            initialize_cluster_params_fn = initialize_cluster_params_multivariate_normal
            likelihood_fn = likelihood_multivariate_normal

            torch_cluster_params = dict(
                means=torch.full(
                    size=(max_num_latents, obs_dim),
                    fill_value=0.,
                    dtype=np.float32),
                stddevs=torch.full(
                    size=(max_num_latents, obs_dim, obs_dim),
                    fill_value=0.,
                    dtype=np.float32))

            likelihood_fit_fn = self.fit_likelihood_normal

        elif self.likelihood_params['distribution'] == 'dirichlet_multinomial':
            initialize_cluster_params_fn = setup_cluster_params_dirichlet_multinomial
            likelihood_fn = likelihood_dirichlet_multinomial

            torch_cluster_params = dict(
                concentrations=torch.full(size=(max_num_latents, obs_dim),
                                          fill_value=np.nan,
                                          dtype=np.float32,
                                          requires_grad=True))

            # fit_fn = self.fit_dirichlet_multinomial

            raise NotImplementedError

        else:
            raise NotImplementedError

        self.fit_results = likelihood_fit_fn(
            torch_observations=torch_observations,
            torch_observations_times=torch_observations_times,
            torch_cluster_params=torch_cluster_params,
            initialize_cluster_params_fn=initialize_cluster_params_fn,
            likelihood_fn=likelihood_fn,
            cluster_assignment_priors=cluster_assignment_priors,
            cluster_assignment_posteriors=cluster_assignment_posteriors,
            cluster_assignment_posteriors_running_sum=cluster_assignment_posteriors_running_sum,
            num_clusters_posteriors=num_cluster_posteriors,
        )

        return self.fit_results

    def fit_likelihood_normal(self,
                              torch_observations: np.ndarray,
                              torch_observations_times: np.ndarray,
                              torch_cluster_params: np.ndarray,
                              initialize_cluster_params_fn: Callable,
                              likelihood_fn: Callable,
                              cluster_assignment_priors: np.ndarray,
                              cluster_assignment_posteriors: np.ndarray,
                              cluster_assignment_posteriors_running_sum: np.ndarray,
                              num_clusters_posteriors: np.ndarray,
                              ):

        for obs_idx, torch_observation in enumerate(torch_observations):

            # Create parameters for each potential new cluster
            initialize_cluster_params_fn(observation=torch_observation,
                                         obs_idx=obs_idx,
                                         torch_cluster_params=torch_cluster_params)

            if obs_idx == 0:
                # first customer has to go at first table
                cluster_assignment_priors[obs_idx, 0] = 1.
                cluster_assignment_posteriors[obs_idx, 0] = 1.
                num_clusters_posteriors[obs_idx, 0] = 1.

                # update running sum of posteriors
                cluster_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                    cluster_assignment_posteriors_running_sum[obs_idx - 1, :],
                    cluster_assignment_posteriors[obs_idx, :])

                self.dynamics.initialize_state(
                    customer_assignment_probs=cluster_assignment_posteriors[obs_idx, :],
                    time=torch_observations_times[obs_idx])

            else:

                # Step 1: Construct prior.
                # Shape: (max num clusters,)
                cluster_assignment_prior = self.dynamics.run_dynamics(
                    time_start=torch_observations_times[obs_idx],
                    time_end=torch_observations_times[obs_idx + 1])['N']

                cluster_assignment_prior2 = cluster_assignment_prior.copy()

                # add new table probability
                cluster_assignment_prior[1:obs_idx + 1] += self.mixing_params['alpha'] * \
                                                           num_clusters_posteriors[obs_idx - 1, :obs_idx].copy()
                # renormalize
                cluster_assignment_prior /= np.sum(cluster_assignment_prior)

                assert_np_no_nan_no_inf_is_real(cluster_assignment_prior)

                # record latent prior
                cluster_assignment_priors[obs_idx, :len(cluster_assignment_prior)] = cluster_assignment_prior

                approx_lower_bounds = []
                for vi_idx in range(self.num_coord_ascent_steps_per_obs):

                    if self.numerically_optimize:
                        raise NotImplementedError
                    else:
                        with torch.no_grad():
                            print(f'Obs Idx: {obs_idx}, VI idx: {vi_idx}')

                    # E step: infer posteriors using parameters
                    likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                        torch_observation=torch_observation,
                        obs_idx=obs_idx,
                        cluster_parameters=cluster_parameters)
                    assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
                    assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

                    if torch.allclose(likelihoods_per_latent, torch.zeros(1)):
                        # print('Complex branch')
                        # we need to deal with numerical instability
                        # the problem is that if log likelihoods are large and negative e.g. -5000, then
                        # the likelihoods will all be 0. Consequently, multiplying the likelihoods and
                        # priors followed by normalizing produces all 0.
                        # the solution is to realize that th
                        table_assignment_log_prior = torch.log(cluster_assignment_prior)
                        table_assignment_log_numerator = torch.add(
                            log_likelihoods_per_latent.detach(),
                            table_assignment_log_prior)
                        max_table_assignment_log_numerator = torch.max(table_assignment_log_numerator)
                        diff_table_assignment_log_numerator = torch.subtract(
                            table_assignment_log_numerator,
                            max_table_assignment_log_numerator)

                        exp_summed_diff_table_assignment_log_numerator = torch.sum(torch.exp(
                            diff_table_assignment_log_numerator))
                        log_normalization = max_table_assignment_log_numerator \
                                            + torch.log(exp_summed_diff_table_assignment_log_numerator)

                        table_assignment_log_posterior = log_likelihoods_per_latent.detach() \
                                                         + table_assignment_log_prior \
                                                         - log_normalization
                        table_assignment_posterior = torch.exp(table_assignment_log_posterior)
                    else:
                        # print('Simple branch')
                        # if no numerical instability, go with the classic
                        # p(z|o, history) = p(o|z)p(z|history)/p(o|history)
                        unnormalized_table_assignment_posterior = torch.multiply(
                            likelihoods_per_latent.detach(),
                            cluster_assignment_prior)
                        table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                            unnormalized_table_assignment_posterior)

                    # sometimes, negative numbers like -3e-84 somehow sneak in. remove
                    table_assignment_posterior[table_assignment_posterior < 0.] = 0.

                    # check that posterior still close to 1.
                    assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

                    # record latent posterior
                    cluster_assignment_posteriors[obs_idx, :len(table_assignment_posterior)] = \
                        table_assignment_posterior.detach().clone()

                    # update running sum of posteriors
                    cluster_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                        cluster_assignment_posteriors_running_sum[obs_idx - 1, :],
                        cluster_assignment_posteriors[obs_idx, :])
                    assert torch.allclose(torch.sum(cluster_assignment_posteriors_running_sum[obs_idx, :]),
                                          torch.Tensor([obs_idx + 1]).double())

                    # M step: update parameters
                    # Note: log likelihood is all we need for optimization because
                    # log p(x, z; params) = log p(x|z; params) + log p(z)
                    # and the second is constant w.r.t. params gradient
                    loss = torch.mean(log_likelihoods_per_latent)
                    loss.backward()

                    # instead of typical dynamics:
                    #       p_k <- p_k + (obs - p_k) / number of obs assigned to kth cluster
                    # we use the new dynamics
                    #       p_k <- p_k + posterior(obs belongs to kth cluster) * (obs - p_k) / total mass on kth cluster
                    # that effectively means the learning rate should be this scaled_prefactor
                    scaled_learning_rate = learning_rate * torch.divide(
                        cluster_assignment_posteriors[obs_idx, :],
                        cluster_assignment_posteriors_running_sum[obs_idx, :]) / num_em_steps
                    scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
                    scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

                    # don't update the newest cluster
                    scaled_learning_rate[obs_idx] = 0.

                    for param_descr, param_tensor in cluster_parameters.items():
                        # the scaled learning rate has shape (num latents,) aka (num obs,)
                        # we need to create extra dimensions of size 1 for broadcasting to work
                        # because param_tensor can have variable number of dimensions e.g. (num obs, obs dim)
                        # for mean vs (num obs, obs dim, obs dim) for covariance, we need to dynamically
                        # add the correct number of dimensions
                        reshaped_scaled_learning_rate = scaled_learning_rate.view(
                            [scaled_learning_rate.shape[0]] + [1 for _ in range(len(param_tensor.shape[1:]))])
                        if param_tensor.grad is None:
                            continue
                        else:
                            scaled_param_tensor_grad = torch.multiply(
                                reshaped_scaled_learning_rate,
                                param_tensor.grad)
                            param_tensor.data += scaled_param_tensor_grad
                            assert_torch_no_nan_no_inf(param_tensor.data[:obs_idx + 1])

                # # previous approach with time complexity O(t^2)
                # # update posterior over number of tables using posterior over customer seat
                # for k1, p_z_t_equals_k1 in enumerate(cluster_assignment_posteriors[obs_idx, :obs_idx + 1]):
                #     for k2, p_prev_num_tables_equals_k2 in enumerate(num_clusters_posteriors[obs_idx - 1, :obs_idx + 1]):
                #         # advance number of tables by 1 if customer seating > number of current tables
                #         if k1 > k2 + 1:
                #             num_clusters_posteriors.data[obs_idx, k2 + 1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                #         # customer allocated to previous table
                #         elif k1 <= k2:
                #             num_clusters_posteriors.data[obs_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                #         # create new table
                #         elif k1 == k2 + 1:
                #             num_clusters_posteriors.data[obs_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                #         else:
                #             raise ValueError
                # assert torch.allclose(torch.sum(num_clusters_posteriors[obs_idx, :]), one_tensor)

                # new approach with time complexity O(t)
                # update posterior over number of tables using posterior over customer seat
                cum_table_assignment_posterior = torch.cumsum(
                    cluster_assignment_posteriors[obs_idx, :obs_idx + 1],
                    dim=0)
                one_minus_cum_table_assignment_posterior = 1. - cum_table_assignment_posterior
                prev_table_posterior = num_clusters_posteriors[obs_idx - 1, :obs_idx]
                num_clusters_posteriors[obs_idx, :obs_idx] += torch.multiply(
                    cum_table_assignment_posterior[:-1],
                    prev_table_posterior)
                num_clusters_posteriors[obs_idx, 1:obs_idx + 1] += torch.multiply(
                    one_minus_cum_table_assignment_posterior[:-1],
                    prev_table_posterior)

                self.dynamics.update_state(
                    customer_assignment_probs=cluster_assignment_posterior,
                    time=torch_observations_times[obs_idx],
                )

        bayesian_recursion_results = dict(
            cluster_assignment_priors=cluster_assignment_priors.numpy(),
            cluster_assignment_posteriors=cluster_assignment_posteriors.numpy(),
            cluster_assignment_posteriors_running_sum=cluster_assignment_posteriors_running_sum.numpy(),
            num_clusters_posteriors=num_clusters_posteriors.numpy(),
            parameters={k: v.detach().numpy() for k, v in torch_cluster_params.items()},
        )

        return bayesian_recursion_results

    def advance_cluster_occupancies(self,
                                    previous_cluster_assignment_posteriors_running_sum: np.ndarray,
                                    curr_time: float,
                                    prev_time: float,
                                    ) -> np.ndarray:

        if self.mixing_params['dynamics_str'] == 'step':
            return previous_cluster_assignment_posteriors_running_sum

        elapsed_time = curr_time - prev_time

        if self.mixing_params['dynamics_str'] == 'exp':
            exp_change = np.exp(- self.mixing_params['dynamics_params']['b'] * elapsed_time
                                / self.mixing_params['dynamics_params']['a'])
            new_cluster_masses = exp_change * previous_cluster_assignment_posteriors_running_sum
        elif self.mixing_params['dynamics_str'] == 'sinusoid':
            raise NotImplementedError
        elif self.mixing_params['dynamics_str'] == 'hyperbolic':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return new_cluster_masses

    def features_after_last_obs(self) -> np.ndarray:
        """
        Returns array of shape (num features, feature dimension)
        """
        raise NotImplementedError


def initialize_cluster_params_multivariate_normal(torch_observation: np.ndarray,
                                                  obs_idx: int,
                                                  cluster_params: Dict[str, np.ndarray]):
    assert_np_no_nan_no_inf_is_real(torch_observation)
    cluster_params['means'].data[obs_idx, :] = torch_observation
    cluster_params['stddevs'].data[obs_idx, :, :] = torch.eye(torch_observation.shape[0])


# Gaussian likelihood setup
def likelihood_multivariate_normal(torch_observation,
                                   obs_idx,
                                   cluster_params):
    obs_dim = torch_observation.shape[0]
    covs = torch.stack([torch.matmul(torch.eye(obs_dim), torch.eye(obs_dim).T)
                        for stddev in cluster_params['stddevs']]).double()

    multivar_normal = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=cluster_params['means'][:obs_idx + 1],
        covariance_matrix=covs[:obs_idx + 1],
    )

    log_likelihoods_per_latent = multivar_normal.log_prob(value=torch_observation)
    likelihoods_per_latent = torch.exp(log_likelihoods_per_latent)

    return likelihoods_per_latent, log_likelihoods_per_latent


# Dirichlet multinomial cluster parameters
def setup_cluster_params_dirichlet_multinomial(observation: np.ndarray,
                                               obs_idx: int,
                                               cluster_parameters: Dict[str, np.ndarray],
                                               epsilon: float = 10.):
    assert_np_no_nan_no_inf_is_real(observation)
    cluster_parameters['concentrations'][obs_idx, :] = observation + epsilon


# Dirichlet multinomial likelihood setup
def likelihood_dirichlet_multinomial(torch_observation,
                                     obs_idx,
                                     cluster_parameters):
    words_in_doc = torch.sum(torch_observation)
    total_concentrations_per_latent = torch.sum(
        cluster_parameters['concentrations'][:obs_idx + 1], dim=1)

    # Intermediate computations
    log_numerator = torch.log(words_in_doc) + log_beta(a=total_concentrations_per_latent, b=words_in_doc)
    log_beta_terms = log_beta(a=cluster_parameters['concentrations'][:obs_idx + 1],
                              b=torch_observation)

    log_x_times_beta_terms = torch.add(log_beta_terms, torch.log(torch_observation))
    log_x_times_beta_terms[torch.isnan(log_x_times_beta_terms)] = 0.
    log_denominator = torch.sum(log_x_times_beta_terms, dim=1)

    assert_np_no_nan_no_inf_is_real(log_denominator)
    log_likelihoods_per_latent = log_numerator - log_denominator

    assert_np_no_nan_no_inf_is_real(log_likelihoods_per_latent)
    likelihoods_per_latent = torch.exp(log_likelihoods_per_latent)

    return likelihoods_per_latent, log_likelihoods_per_latent
