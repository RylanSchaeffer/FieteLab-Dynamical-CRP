import numpy as np
import sklearn.mixture
from typing import Dict

from rncrp.inference.base import BaseModel


class RecursiveNonstationaryCRP(BaseModel):
    """
    Variational Inference for Dirichlet Process Gaussian Mixture Model, as
    proposed by Blei and Jordan (2006).

    Wrapper around scikit-learn's implementation.
    """

    def __init__(self,
                 gen_model_params: Dict[str, Dict[str, float]],
                 model_str: str = 'RN-CRP',
                 plot_dir: str = None,
                 **kwargs,
                 ):
        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.feature_prior_params = gen_model_params['feature_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        self.model_str = model_str
        self.plot_dir = plot_dir
        self.fit_results = None

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray):

        num_obs, obs_dim = observations.shape
        max_num_latents = num_obs

        cluster_assignment_priors = np.zeros((num_obs, max_num_latents),
                                             dtype=np.float32)
        cluster_assignment_priors[0, 0] = 1.

        cluster_assignment_posteriors = np.zeros((num_obs, max_num_latents),
                                                 dtype=np.float32, )

        cluster_assignment_posteriors_running_sum = np.zeros((num_obs, max_num_latents),
                                                             dtype=np.float32)

        num_cluster_posteriors = np.zeros((num_obs, max_num_latents),
                                          dtype=np.float32)

        if likelihood_model == 'multivariate_normal':
            setup_cluster_params_fn = setup_cluster_params_multivariate_normal
            likelihood_fn = likelihood_multivariate_normal

            cluster_params = dict(
                means=np.full(
                    size=(max_num_latents, obs_dim),
                    fill_value=0.,
                    dtype=np.float32),
                stddevs=np.full(
                    size=(max_num_latents, obs_dim, obs_dim),
                    fill_value=0.,
                    dtype=np.float32))

        elif likelihood_model == 'dirichlet_multinomial':
            setup_cluster_params_fn = setup_cluster_params_dirichlet_multinomial
            likelihood_fn = likelihood_dirichlet_multinomial

            cluster_params = dict(concentrations=torch.full(size=(max_num_latents, obs_dim),
                                                            fill_value=np.nan,
                                                            dtype=torch.float64,
                                                            requires_grad=True))

        # todo: set up other likelihoods as needed
        else:
            raise NotImplementedError

        self.fit_results = self.fit_likelihood_normal(
            observations=observations,
            observations_times=observations_times,
            cluster_assignment_priors=cluster_assignment_priors,
            cluster_assignment_posteriors=cluster_assignment_posteriors,
            cluster_assignment_posteriors_running_sum=cluster_assignment_posteriors_running_sum,
            num_clusters_posteriors=num_cluster_posteriors,
        )

        return self.fit_results

    def fit_likelihood_normal(self,
                              observations: np.ndarray,
                              observations_times: np.ndarray,
                              cluster_assignment_priors: np.ndarray,
                              cluster_assignment_posteriors: np.ndarray,
                              cluster_assignment_posteriors_running_sum: np.ndarray,
                              num_clusters_posteriors: np.ndarray,
                              ):

        for obs_idx, (observation, observation_time) in enumerate(observations,
                                                                  observations_times):

            # Create parameters for each potential new cluster
            setup_cluster_params_fn(torch_observation=observation,
                                    obs_idx=obs_idx,
                                    cluster_params=cluster_params)

            if obs_idx == 0:
                # first customer has to go at first table
                cluster_assignment_priors[obs_idx, 0] = 1.
                cluster_assignment_posteriors[obs_idx, 0] = 1.
                num_clusters_posteriors[obs_idx, 0] = 1.

                # update running sum of posteriors
                cluster_assignment_posteriors_running_sum[obs_idx, :] = np.add(
                    cluster_assignment_posteriors_running_sum[obs_idx - 1, :],
                    cluster_assignment_posteriors[obs_idx, :])
            else:

                # construct prior
                table_assignment_prior = torch.clone(
                    table_assignment_posteriors_running_sum[obs_idx - 1, :obs_idx + 1])
                # we don't subtract 1 because Python uses 0-based indexing
                assert torch.allclose(torch.sum(table_assignment_prior), torch.Tensor([obs_idx]).double())
                # add new table probability
                table_assignment_prior[1:] += concentration_param * torch.clone(
                    num_table_posteriors[obs_idx - 1, :obs_idx])
                # renormalize
                table_assignment_prior /= (concentration_param + obs_idx)

                # sometimes, negative numbers like -3e-84 somehow sneak in. remove
                table_assignment_prior[table_assignment_prior < 0.] = 0.

                assert torch.allclose(torch.sum(table_assignment_prior), one_tensor)
                assert_torch_no_nan_no_inf(table_assignment_prior)

                # record latent prior
                table_assignment_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior

                for em_idx in range(num_em_steps):

                    optimizer.zero_grad()

                    # E step: infer posteriors using parameters
                    likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                        torch_observation=observation,
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
                        table_assignment_log_prior = torch.log(table_assignment_prior)
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
                            table_assignment_prior)
                        table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                            unnormalized_table_assignment_posterior)

                    # sometimes, negative numbers like -3e-84 somehow sneak in. remove
                    table_assignment_posterior[table_assignment_posterior < 0.] = 0.

                    # check that posterior still close to 1.
                    assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

                    # record latent posterior
                    table_assignment_posteriors[obs_idx, :len(table_assignment_posterior)] = \
                        table_assignment_posterior.detach().clone()

                    # update running sum of posteriors
                    table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                        table_assignment_posteriors_running_sum[obs_idx - 1, :],
                        table_assignment_posteriors[obs_idx, :])
                    assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
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
                        table_assignment_posteriors[obs_idx, :],
                        table_assignment_posteriors_running_sum[obs_idx, :]) / num_em_steps
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
                # for k1, p_z_t_equals_k1 in enumerate(table_assignment_posteriors[obs_idx, :obs_idx + 1]):
                #     for k2, p_prev_num_tables_equals_k2 in enumerate(num_table_posteriors[obs_idx - 1, :obs_idx + 1]):
                #         # advance number of tables by 1 if customer seating > number of current tables
                #         if k1 > k2 + 1:
                #             num_table_posteriors.data[obs_idx, k2 + 1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                #         # customer allocated to previous table
                #         elif k1 <= k2:
                #             num_table_posteriors.data[obs_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                #         # create new table
                #         elif k1 == k2 + 1:
                #             num_table_posteriors.data[obs_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                #         else:
                #             raise ValueError
                # assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)

                # new approach with time complexity O(t)
                # update posterior over number of tables using posterior over customer seat
                cum_table_assignment_posterior = torch.cumsum(
                    table_assignment_posteriors[obs_idx, :obs_idx + 1],
                    dim=0)
                one_minus_cum_table_assignment_posterior = 1. - cum_table_assignment_posterior
                prev_table_posterior = num_table_posteriors[obs_idx - 1, :obs_idx]
                num_table_posteriors[obs_idx, :obs_idx] += torch.multiply(
                    cum_table_assignment_posterior[:-1],
                    prev_table_posterior)
                num_table_posteriors[obs_idx, 1:obs_idx + 1] += torch.multiply(
                    one_minus_cum_table_assignment_posterior[:-1],
                    prev_table_posterior)
                assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)

        # TODO: investigate how cluster parameters fall below initialization
        # is gradient descent not correct?
        # check that likelihood is maximized. Am I minimizing the likelihood? Where does the negative
        # appear?   a
        bayesian_recursion_results = dict(
            table_assignment_priors=table_assignment_priors.numpy(),
            table_assignment_posteriors=table_assignment_posteriors.numpy(),
            table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.numpy(),
            num_table_posteriors=num_table_posteriors.numpy(),
            parameters={k: v.detach().numpy() for k, v in cluster_parameters.items()},
        )

        return bayesian_recursion_results
