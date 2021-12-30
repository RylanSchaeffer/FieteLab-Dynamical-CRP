import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import os
import torch
from timeit import default_timer as timer

import helpers.numpy
import helpers.torch
import utils.inference
import utils.metrics 
import utils.plot
from utils.single_run import *
from data.synthetic import *
torch.set_default_tensor_type('torch.FloatTensor')

def sweep_parameters(plot_dir, sweep_setting):
    assert sweep_setting in {'sweep_dimensions', 'sweep_means', 'sweep_means_anisotropy'}
    
    num_datasets = 10

    # Sweep over number of dimensions
    if sweep_setting == 'sweep_dimensions':
        for dim in np.arange(2, 20, 3):
            plot_dir += '/dim_'
            plot_dir += str(dim)

            inference_algs_results_by_dataset_idx = {}
            sampled_gaussian_data_by_dataset_idx = {}

            for dataset_idx in range(num_datasets):
                print(f'Dataset Index: {dataset_idx}')
                dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
                os.makedirs(dataset_dir, exist_ok=True)

                sampled_gaussian_data = sample_from_mixture_of_gaussians(
                    seq_len=100,
                    num_gaussians=3,
                    gaussian_dim=dim,
                    gaussian_params=dict(gaussian_cov_scaling=0.3,
                                         gaussian_mean_prior_cov_scaling=6.),
                    anisotropy=False)

                dataset_inference_algs_results, dataset_sampled_mix_of_gaussians_results = single_run(
                    dataset_dir=dataset_dir, 
                    sampled_data=sampled_gaussian_data,
                    setting='gaussian')

                inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results
                sampled_gaussian_data_by_dataset_idx[dataset_idx] = dataset_sampled_mix_of_gaussians_results

            utils.plot.plot_inference_algs_comparison(
                plot_dir=plot_dir,
                inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
                dataset_by_dataset_idx=sampled_gaussian_data_by_dataset_idx)

    # Sweep over spread of cluster means
    elif sweep_setting == 'sweep_means':
        for spread in np.arange(3., 20., 4.):
            plot_dir += '/spread_'
            plot_dir += str(spread)

            inference_algs_results_by_dataset_idx = {}
            sampled_gaussian_data_by_dataset_idx = {}

            for dataset_idx in range(num_datasets):
                print(f'Dataset Index: {dataset_idx}')
                dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
                os.makedirs(dataset_dir, exist_ok=True)
                
                sampled_gaussian_data = sample_from_mixture_of_gaussians(
                    seq_len=100,
                    num_gaussians=3,
                    gaussian_dim=2,
                    gaussian_params=dict(gaussian_cov_scaling=0.3,
                                         gaussian_mean_prior_cov_scaling=spread),
                    anisotropy=False)

                dataset_inference_algs_results, dataset_sampled_mix_of_gaussians_results = single_run(
                    dataset_dir=dataset_dir, 
                    sampled_data=sampled_gaussian_data,
                    setting='gaussian')

                inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results
                sampled_gaussian_data_by_dataset_idx[dataset_idx] = dataset_sampled_mix_of_gaussians_results

            utils.plot.plot_inference_algs_comparison(
                plot_dir=plot_dir,
                inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
                dataset_by_dataset_idx=sampled_gaussian_data_by_dataset_idx)

    # Sweep over anisotropy of cluster means
    elif sweep_setting == 'sweep_means_anisotropy':
        inference_algs_results_by_dataset_idx = {}
        sampled_gaussian_data_by_dataset_idx = {}

        for dataset_idx in range(num_datasets):
            print(f'Dataset Index: {dataset_idx}')
            dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
            os.makedirs(dataset_dir, exist_ok=True)
            
            sampled_gaussian_data = sample_from_mixture_of_gaussians(
                seq_len=100,
                num_gaussians=3,
                gaussian_dim=2,
                gaussian_params=dict(gaussian_cov_scaling=0.3,
                                     gaussian_mean_prior_cov_scaling=6.),
                anisotropy=True)

            dataset_inference_algs_results, dataset_sampled_mix_of_gaussians_results = single_run(
                dataset_dir=dataset_dir, 
                sampled_data=sampled_gaussian_data,
                setting='gaussian')

            inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results
            sampled_gaussian_data_by_dataset_idx[dataset_idx] = dataset_sampled_mix_of_gaussians_results

        utils.plot.plot_inference_algs_comparison(
            plot_dir=plot_dir,
            inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
            dataset_by_dataset_idx=sampled_gaussian_data_by_dataset_idx)


def main():
    plot_dir = '0-gaussian_mixture/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    # Number of dimensions
    sweep_parameters(plot_dir+'/sweep_dimensions','sweep_dimensions')

    # Spread of cluster means
    sweep_parameters(plot_dir+'/sweep_means','sweep_means')

    # Anisotropy of cluster means
    sweep_parameters(plot_dir+'/sweep_means_anisotropy','sweep_means_anisotropy')


if __name__ == '__main__':
    main()


# def dynamical_crp(observations,
#                   observation_times,
#                   inference_alg_params: Dict[str, float],
#                   inference_dynamics_str: str,
#                   likelihood_model: str,
#                   learning_rate,
#                   num_em_steps: int = 3):
#
#     assert inference_alg_params['alpha'] > 0.
#     # TODO: implement beta
#     assert inference_alg_params['beta'] == 0.
#     assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial',
#                                 'bernoulli', 'continuous_bernoulli'}
#     num_obs, obs_dim = observations.shape
#
#     # The recursion does not require recording the full history of priors/posteriors
#     # but we record the full history for subsequent analysis
#     max_num_latents = num_obs
#     table_assignment_priors = torch.zeros(
#         (num_obs, max_num_latents),
#         dtype=torch.float64,
#         requires_grad=False)
#     table_assignment_priors[0, 0] = 1.
#
#     table_assignment_posteriors = torch.zeros(
#         (num_obs, max_num_latents),
#         dtype=torch.float64,
#         requires_grad=False)
#
#     table_assignment_posteriors_running_sum = torch.zeros(
#         (num_obs, max_num_latents),
#         dtype=torch.float64,
#         requires_grad=False)
#
#     num_table_posteriors = torch.zeros(
#         (num_obs, max_num_latents),
#         dtype=torch.float64,
#         requires_grad=False)
#
#     if likelihood_model == 'continuous_bernoulli':
#         # need to use logits, otherwise gradient descent will carry parameters outside
#         # valid interval
#         cluster_parameters = dict(
#             logits=torch.full(
#                 size=(max_num_latents, obs_dim),
#                 fill_value=np.nan,
#                 dtype=torch.float64,
#                 requires_grad=True)
#         )
#         create_new_cluster_params_fn = create_new_cluster_params_continuous_bernoulli
#         likelihood_fn = likelihood_continuous_bernoulli
#
#         # make sure no observation is 0 or 1 by adding epsilon
#         epsilon = 1e-2
#         observations[observations == 1.] -= epsilon
#         observations[observations == 0.] += epsilon
#     elif likelihood_model == 'dirichlet_multinomial':
#         cluster_parameters = dict(
#             topics_concentrations=torch.full(
#                 size=(max_num_latents, obs_dim),
#                 fill_value=np.nan,
#                 dtype=torch.float64,
#                 requires_grad=True),
#         )
#         create_new_cluster_params_fn = create_new_cluster_params_dirichlet_multinomial
#         likelihood_fn = likelihood_dirichlet_multinomial
#     elif likelihood_model == 'multivariate_normal':
#         cluster_parameters = dict(
#             means=torch.full(
#                 size=(max_num_latents, obs_dim),
#                 fill_value=np.nan,
#                 dtype=torch.float64,
#                 requires_grad=True),
#             stddevs=torch.full(
#                 size=(max_num_latents, obs_dim, obs_dim),
#                 fill_value=np.nan,
#                 dtype=torch.float64,
#                 requires_grad=True),
#         )
#         create_new_cluster_params_fn = create_new_cluster_params_multivariate_normal
#         likelihood_fn = likelihood_multivariate_normal
#     else:
#         raise NotImplementedError
#
#     optimizer = torch.optim.SGD(params=cluster_parameters.values(), lr=1.)
#
#     # create dynamics
#     dynamics = utils.dynamics.dynamics_factory(
#         inference_dynamics_str=inference_dynamics_str)
#
#     # needed later for error checking
#     one_tensor = torch.Tensor([1.]).double()
#
#     alpha = inference_alg_params['alpha']
#     beta = inference_alg_params['beta']
#
#     torch_observations = torch.from_numpy(observations)
#     torch_observation_times = torch.from_numpy(observation_times)
#     for obs_idx, (torch_observation, torch_observation_time) in \
#             enumerate(zip(torch_observations, torch_observation_times)):
#
#         # create new params for possible cluster, centered at that point
#         create_new_cluster_params_fn(
#             torch_observation=torch_observation,
#             obs_idx=obs_idx,
#             cluster_parameters=cluster_parameters)
#
#         if obs_idx == 0:
#             # first customer has to go at first table
#             table_assignment_priors[obs_idx, 0] = 1.
#             table_assignment_posteriors[obs_idx, 0] = 1.
#             num_table_posteriors[obs_idx, 0] = 1.
#
#             dynamics.initialize_state(
#                 customer_assignment_probs=table_assignment_posteriors[obs_idx, :].numpy(),
#                 time=torch_observation_time.numpy())
#
#             # update running sum of posteriors
#             table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
#                 table_assignment_posteriors_running_sum[obs_idx - 1, :],
#                 table_assignment_posteriors[obs_idx, :])
#             assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
#                                   torch.Tensor([obs_idx + 1]).double())
#
#         else:
#             # construct prior
#             state = dynamics.run_dynamics(
#                 time_start=observation_times[obs_idx - 1],
#                 time_end=observation_times[obs_idx])
#             state = torch.from_numpy(state['N'][:obs_idx + 1]).clone()
#             # interestingly, creating torch.tensor from np.ndarray and then modifying the tensor
#             # results in the np.ndarray being modified
#             table_assignment_prior = state.clone()
#
#             # check against previous approach
#             # table_assignment_prior2 = torch.clone(
#             #     table_assignment_posteriors_running_sum[obs_idx - 1, :obs_idx + 1])
#             # assert torch.allclose(table_assignment_prior, table_assignment_prior2)
#
#             # check for correctness if possible
#             if inference_dynamics_str == 'perfectintegrator':
#                 assert torch.allclose(torch.sum(table_assignment_prior),
#                                       torch.Tensor([obs_idx]).double())
#
#             # add new table term then normalize
#             table_assignment_prior[1:] += alpha * torch.clone(
#                 num_table_posteriors[obs_idx - 1, :obs_idx])
#             table_assignment_prior /= (alpha + torch.sum(state))
#
#             # sometimes, negative numbers like -3e-84 somehow sneak in. remove.
#             table_assignment_prior[table_assignment_prior < 0.] = 0.
#
#             assert torch.allclose(torch.sum(table_assignment_prior), one_tensor)
#             utils.helpers.assert_torch_no_nan_no_inf(table_assignment_prior)
#
#             # record latent prior
#             table_assignment_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior
#
#             for em_idx in range(num_em_steps):
#
#                 optimizer.zero_grad()
#
#                 # E step: infer posteriors using parameters
#                 likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
#                     torch_observation=torch_observation,
#                     obs_idx=obs_idx,
#                     cluster_parameters=cluster_parameters)
#                 assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
#                 assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))
#
#                 if torch.allclose(likelihoods_per_latent, torch.zeros(1)):
#                     # print('Complex branch')
#                     # we need to deal with numerical instability
#                     # the problem is that if log likelihoods are large and negative e.g. -5000, then
#                     # the likelihoods will all be 0. Consequently, multiplying the likelihoods and
#                     # priors followed by normalizing produces all 0.
#                     # the solution is to realize that th
#                     table_assignment_log_prior = torch.log(table_assignment_prior)
#                     table_assignment_log_numerator = torch.add(
#                         log_likelihoods_per_latent.detach(),
#                         table_assignment_log_prior)
#                     max_table_assignment_log_numerator = torch.max(table_assignment_log_numerator)
#                     diff_table_assignment_log_numerator = torch.subtract(
#                         table_assignment_log_numerator,
#                         max_table_assignment_log_numerator)
#
#                     exp_summed_diff_table_assignment_log_numerator = torch.sum(torch.exp(
#                         diff_table_assignment_log_numerator))
#                     log_normalization = max_table_assignment_log_numerator\
#                                         + torch.log(exp_summed_diff_table_assignment_log_numerator)
#
#                     table_assignment_log_posterior = log_likelihoods_per_latent.detach()\
#                                                      + table_assignment_log_prior\
#                                                      - log_normalization
#                     table_assignment_posterior = torch.exp(table_assignment_log_posterior)
#                 else:
#                     # print('Simple branch')
#                     # if no numerical instability, go with the classic
#                     # p(z|o, history) = p(o|z)p(z|history)/p(o|history)
#                     unnormalized_table_assignment_posterior = torch.multiply(
#                         likelihoods_per_latent.detach(),
#                         table_assignment_prior)
#                     table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
#                         unnormalized_table_assignment_posterior)
#
#                 # sometimes, negative numbers like -3e-84 somehow sneak in. remove.
#                 table_assignment_posterior[table_assignment_posterior < 0.] = 0.
#
#                 # TODO: possible error if prior probability of new cluster is 0, but all other likelihoods are 0
#                 # check that posterior still close to 1.
#                 assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)
#
#                 # import matplotlib.pyplot as plt
#                 # plt.scatter(cluster_parameters['means'].detach().numpy()[:, 0],
#                 #          cluster_parameters['means'].detach().numpy()[:, 1],
#                 #             label='centroids')
#                 # plt.scatter(torch_observation.detach().numpy()[0],
#                 #          torch_observation.detach().numpy()[1],
#                 #             label='new')
#                 # plt.legend()
#                 # plt.show()
#
#                 # record latent posterior
#                 table_assignment_posteriors[obs_idx, :len(table_assignment_posterior)] = \
#                     table_assignment_posterior.detach().clone()
#
#                 # update running sum of posteriors
#                 if inference_dynamics_str == 'perfectintegrator':
#                     table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
#                         table_assignment_posteriors_running_sum[obs_idx - 1, :],
#                         table_assignment_posteriors[obs_idx, :])
#                     assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
#                                           torch.Tensor([obs_idx + 1]).double())
#
#                 # M step: update parameters
#                 # Note: log likelihood is all we need for optimization because
#                 # log p(x, z; params) = log p(x|z; params) + log p(z)
#                 # and the second is constant w.r.t. params gradient
#                 loss = torch.mean(log_likelihoods_per_latent)
#                 loss.backward()
#
#                 # instead of typical dynamics:
#                 #       p_k <- p_k + (obs - p_k) / number of obs assigned to kth cluster
#                 # we use the new dynamics
#                 #       p_k <- p_k + posterior(obs belongs to kth cluster) * (obs - p_k) / total mass on kth cluster
#                 # that effectively means the learning rate should be this scaled_prefactor
#                 scaled_learning_rate = learning_rate * torch.divide(
#                     table_assignment_posteriors[obs_idx, :],
#                     table_assignment_posteriors_running_sum[obs_idx, :]) / num_em_steps
#                 scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
#                 scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.
#
#                 # don't update the newest cluster
#                 scaled_learning_rate[obs_idx] = 0.
#
#                 for param_descr, param_tensor in cluster_parameters.items():
#                     # the scaled learning rate has shape (num latents,) aka (num obs,)
#                     # we need to create extra dimensions of size 1 for broadcasting to work
#                     # because param_tensor can have variable number of dimensions e.g. (num obs, obs dim)
#                     # for mean vs (num obs, obs dim, obs dim) for covariance, we need to dynamically
#                     # add the correct number of dimensions
#                     reshaped_scaled_learning_rate = scaled_learning_rate.view(
#                         [scaled_learning_rate.shape[0]] + [1 for _ in range(len(param_tensor.shape[1:]))])
#                     if param_tensor.grad is None:
#                         continue
#                     else:
#                         scaled_param_tensor_grad = torch.multiply(
#                             reshaped_scaled_learning_rate,
#                             param_tensor.grad)
#                         param_tensor.data += scaled_param_tensor_grad
#                         utils.helpers.assert_torch_no_nan_no_inf(param_tensor.data[:obs_idx + 1])
#
#             state = dynamics.update_state(
#                 customer_assignment_probs=table_assignment_posteriors[obs_idx, :].numpy(),
#                 time=torch_observation_time.numpy())
#
#             if inference_dynamics_str == 'perfect_integrator':
#                 assert torch.allclose(table_assignment_posteriors_running_sum[obs_idx, :],
#                                       torch.from_numpy(state['N']))
#
#             # new approach with time complexity O(t)
#             # update posterior over number of tables using posterior over customer seat
#             cum_table_assignment_posterior = torch.cumsum(
#                 table_assignment_posteriors[obs_idx, :obs_idx + 1],
#                 dim=0)
#             one_minus_cum_table_assignment_posterior = 1. - cum_table_assignment_posterior
#             prev_table_posterior = num_table_posteriors[obs_idx - 1, :obs_idx]
#             num_table_posteriors[obs_idx, :obs_idx] += torch.multiply(
#                 cum_table_assignment_posterior[:-1],
#                 prev_table_posterior)
#             num_table_posteriors[obs_idx, 1:obs_idx + 1] += torch.multiply(
#                 one_minus_cum_table_assignment_posterior[:-1],
#                 prev_table_posterior)
#             assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)
#
#     # TODO: investigate how cluster parameters fall below initialization for Dirichlet Multinomial
#     # is gradient descent not correct?
#     # check that likelihood is maximized. Am I minimizing the likelihood? Where does the negative
#     # appear?
#     bayesian_recursion_results = dict(
#         table_assignment_priors=table_assignment_priors.numpy(),
#         table_assignment_posteriors=table_assignment_posteriors.numpy(),
#         table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.numpy(),
#         num_table_posteriors=num_table_posteriors.numpy(),
#         cluster_parameters={k: v.detach().numpy() for k, v in cluster_parameters.items()},
#     )
#
#     return bayesian_recursion_results



# def variational_bayes(observations,
#                       likelihood_model: str,
#                       learning_rate: float,
#                       concentration_param: float,
#                       max_iter: int = 8,  # same as DP-Means
#                       num_initializations: int = 1):
#     # Variational Inference for Dirichlet Process Mixtures
#     # Blei and Jordan (2006)
#     # likelihood_model not used
#     # learning rate not used
#
#     assert concentration_param > 0
#
#     num_obs, obs_dim = observations.shape
#     var_dp_gmm = sklearn.mixture.BayesianGaussianMixture(
#         n_components=num_obs,
#         max_iter=max_iter,
#         n_init=num_initializations,
#         init_params='random',
#         weight_concentration_prior_type='dirichlet_process',
#         weight_concentration_prior=concentration_param)
#     var_dp_gmm.fit(observations)
#     table_assignment_posteriors = var_dp_gmm.predict_proba(observations)
#     table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
#                                                         axis=0)
#     params = dict(means=var_dp_gmm.means_,
#                   covs=var_dp_gmm.covariances_)
#
#     # returns classes assigned and centroids of corresponding classes
#     variational_results = dict(
#         table_assignment_posteriors=table_assignment_posteriors,
#         table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
#         parameters=params,
#     )
#     return variational_results


