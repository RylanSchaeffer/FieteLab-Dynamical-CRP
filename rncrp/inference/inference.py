# import logging
# import numpyro
# import scipy
import numpy as np
import sklearn.mixture
import torch
import torch.distributions
import torch.optim
from helpers.torch_helpers import assert_no_nan_no_inf_is_real


# torch.set_default_tensor_type('torch.DoubleTensor')

# Gaussian cluster parameters


# RN-CRP
def rn_crp(observations,
           concentration_param: float,
           likelihood_model: str,
           learning_rate,
           num_em_steps: int = 3):

    assert concentration_param > 0
    assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial'}  # todo: add more for other settings

    num_obs, obs_dim = observations.shape
    max_num_latents = num_obs

    cluster_assgt_priors = torch.zeros((num_obs, max_num_latents),
                                       dtype=torch.float64,
                                       requires_grad=False)
    cluster_assgt_priors[0, 0] = 1.

    cluster_assgt_posteriors = torch.zeros((num_obs, max_num_latents),
                                           dtype=torch.float64,
                                           requires_grad=False)

    cluster_assgt_posteriors_running_sum = torch.zeros((num_obs, max_num_latents),
                                                       dtype=torch.float64,
                                                       requires_grad=False)

    num_cluster_posteriors = torch.zeros((num_obs, max_num_latents),
                                         dtype=torch.float64,
                                         requires_grad=False)

    optimizer = torch.optim.SGD(params=cluster_params.values(), lr=1.)
    one = torch.Tensor([1.]).double()
    torch_observations = torch.from_numpy(observations)

    def time_kernel(n_prime, n):
        return np.exp(n_prime - n)

    def normalizing_constant(n):
        return concentration_param + sum(map(time_kernel, range(1, n + 1), n * np.ones(n)))

    for obs_idx, torch_observation in enumerate(torch_observations):
        # Create parameters for each potential new cluster
        setup_cluster_params_fn(torch_observation=torch_observation,
                                obs_idx=obs_idx,
                                cluster_params=cluster_params)

        # First customer to first table
        if obs_idx == 0:
            # Construct prior
            cluster_assgt_priors[obs_idx, 0] = 1.
            cluster_assgt_posteriors[obs_idx, 0] = 1.
            num_cluster_posteriors[obs_idx, 0] = 1.

            # Update running sum of posteriors
            # todo: check the math here
            cluster_assgt_posteriors_running_sum[obs_idx, :] = torch.add(
                torch.multiply(
                    cluster_assgt_posteriors_running_sum[obs_idx - 1, :],
                    torch.from_numpy(
                        np.array(map(time_kernel, range(obs_idx), obs_idx * np.ones(obs_idx))).reshape(-1, 1)),
                ),
                concentration_param * cluster_assgt_posteriors[obs_idx,
                                      :])  # todo: not sure if need concentration_param here?

        # Address additional customers
        else:
            # Construct prior
            cluster_assgt_prior = torch.multiply(
                torch.clone(
                    cluster_assgt_posteriors_running_sum[obs_idx - 1, :obs_idx + 1]),
                torch.from_numpy(np.array(map(time_kernel, range(obs_idx + 1), obs_idx * np.ones(obs_idx + 1)))),
            )

            ## Add probability of new table
            cluster_assgt_prior[1:] += concentration_param * torch.clone(num_cluster_posteriors[obs_idx - 1, :obs_idx])
            cluster_assgt_prior /= normalizing_constant(obs_idx)  # renormalize
            cluster_assgt_prior[cluster_assgt_prior < 0.] = 0.  # ensure no negative values present

            assert torch.allclose(torch.sum(cluster_assgt_prior), one)
            assert_no_nan_no_inf_is_real(cluster_assgt_prior)

            # Record latent prior
            cluster_assgt_priors[obs_idx, :len(cluster_assgt_prior)] = cluster_assgt_prior

            for em_idx in range(num_em_steps):
                optimizer.zero_grad()

                # E step: infer posteriors using parameters
                likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                    torch_observation=torch_observation,
                    obs_idx=obs_idx,
                    cluster_params=cluster_params)

                assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
                assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

                if torch.allclose(likelihoods_per_latent, torch.zeros(1)):

                    cluster_assgt_log_prior = torch.log(cluster_assgt_prior)
                    cluster_assgt_log_numerator = torch.add(
                        log_likelihoods_per_latent.detach(),
                        cluster_assgt_log_prior)
                    max_cluster_assgt_log_numerator = torch.max(cluster_assgt_log_numerator)
                    diff_cluster_assgt_log_numerator = torch.subtract(
                        cluster_assgt_log_numerator,
                        max_cluster_assgt_log_numerator)

                    exp_summed_diff_cluster_assgt_log_numerator = torch.sum(torch.exp(
                        diff_cluster_assgt_log_numerator))
                    log_normalization = max_cluster_assgt_log_numerator \
                                        + torch.log(exp_summed_diff_cluster_assgt_log_numerator)

                    cluster_assgt_log_posterior = log_likelihoods_per_latent.detach() \
                                                  + cluster_assgt_log_prior \
                                                  - log_normalization
                    cluster_assgt_posterior = torch.exp(cluster_assgt_log_posterior)
                else:
                    unnormalized_cluster_assgt_posterior = torch.multiply(
                        likelihoods_per_latent.detach(),
                        cluster_assgt_prior)
                    # Normalize posterior
                    cluster_assgt_posterior = unnormalized_cluster_assgt_posterior / torch.sum(
                        unnormalized_cluster_assgt_posterior)

                assert torch.allclose(torch.sum(cluster_assgt_posterior), one)
                cluster_assgt_posterior[cluster_assgt_posterior < 0.] = 0.

                # Record latent posterior
                cluster_assgt_posteriors[obs_idx, :len(cluster_assgt_posterior)] = \
                    cluster_assgt_posterior.detach().clone()

                # Update running sum of posteriors
                # todo: check math here (same as above)
                cluster_assgt_posteriors_running_sum[obs_idx, :] = torch.add(
                    torch.multiply(
                        cluster_assgt_posteriors_running_sum[obs_idx - 1, :],
                        torch.from_numpy(
                            np.array(map(time_kernel, range(obs_idx), obs_idx * np.ones(obs_idx))).reshape(-1, 1)),
                    ),
                    concentration_param * cluster_assgt_posteriors[obs_idx, :])
                assert torch.allclose(torch.sum(cluster_assgt_posteriors_running_sum[obs_idx, :]),
                                      torch.Tensor([obs_idx + 1]).double())

                # M step: update parameters
                loss = torch.mean(log_likelihoods_per_latent)
                loss.backward()

                scaled_learning_rate = learning_rate * torch.divide(
                    cluster_assgt_posteriors[obs_idx, :],
                    cluster_assgt_posteriors_running_sum[obs_idx, :]) / num_em_steps
                scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
                scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

                scaled_learning_rate[obs_idx] = 0.

                for param_descr, param_tensor in cluster_params.items():
                    reshaped_scaled_learning_rate = scaled_learning_rate.view(
                        [scaled_learning_rate.shape[0]] + [1 for _ in range(len(param_tensor.shape[1:]))])
                    if param_tensor.grad is None:
                        continue
                    else:
                        scaled_param_tensor_grad = torch.multiply(
                            reshaped_scaled_learning_rate,
                            param_tensor.grad)
                        param_tensor.data += scaled_param_tensor_grad
                        assert_no_nan_no_inf_is_real(param_tensor.data[:obs_idx + 1])

            cumul_cluster_assgt_posterior = torch.cumsum(
                cluster_assgt_posteriors[obs_idx, :obs_idx + 1],
                dim=0)
            one_minus_cumul_cluster_assgt_posterior = 1. - cumul_cluster_assgt_posterior

            prev_cluster_posterior = num_cluster_posteriors[obs_idx - 1, :obs_idx]

            num_cluster_posteriors[obs_idx, :obs_idx] += torch.multiply(
                cum_cluster_assgt_posterior[:-1],
                prev_cluster_posterior)
            num_cluster_posteriors[obs_idx, 1:obs_idx + 1] += torch.multiply(
                one_minus_cumul_cluster_assgt_posterior[:-1],
                prev_cluster_posterior)
            assert torch.allclose(torch.sum(num_cluster_posteriors[obs_idx, :]), one)

    bayesian_recursion_results = dict(
        cluster_assgt_priors=cluster_assgt_priors.numpy(),
        cluster_assgt_posteriors=cluster_assgt_posteriors.numpy(),
        cluster_assgt_posteriors_running_sum=cluster_assgt_posteriors_running_sum.numpy(),
        num_cluster_posteriors=num_cluster_posteriors.numpy(),
        parameters={k: v.detach().numpy() for k, v in cluster_params.items()},
    )

    return bayesian_recursion_results


