# import logging
# import numpyro
# import scipy
import numpy as np
import sklearn.mixture
import torch
import torch.distributions
import torch.optim
import rncrp.prob_models
from rncrp.helpers.torch import assert_no_nan_no_inf_is_real


# torch.set_default_tensor_type('torch.DoubleTensor')

# Gaussian cluster parameters
def setup_cluster_params_multivariate_normal(torch_observation,
                                             obs_idx,
                                             cluster_params):
    assert_no_nan_no_inf_is_real(torch_observation)
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
def setup_cluster_params_dirichlet_multinomial(torch_observation,
                                               obs_idx,
                                               cluster_parameters):
    assert_no_nan_no_inf_is_real(torch_observation)
    epsilon = 10.  # May need to change
    cluster_parameters['concentrations'].data[obs_idx, :] = torch_observation + epsilon


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

    assert_no_nan_no_inf_is_real(log_denominator)
    log_likelihoods_per_latent = log_numerator - log_denominator

    assert_no_nan_no_inf_is_real(log_likelihoods_per_latent)
    likelihoods_per_latent = torch.exp(log_likelihoods_per_latent)

    return likelihoods_per_latent, log_likelihoods_per_latent


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

    if likelihood_model == 'multivariate_normal':
        setup_cluster_params_fn = setup_cluster_params_multivariate_normal
        likelihood_fn = likelihood_multivariate_normal

        cluster_params = dict(means=torch.full(
            size=(max_num_latents, obs_dim),
            fill_value=0.,
            dtype=torch.float64,
            requires_grad=True),
            stddevs=torch.full(
                size=(max_num_latents, obs_dim, obs_dim),
                fill_value=0.,
                dtype=torch.float64,
                requires_grad=True))

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


# DP-Means
def dp_means(observations,
             concentration_param: float,
             likelihood_model: str,
             learning_rate: float,
             num_passes: int):  # "online" if if num_passes = 1; "offline" if num_passes > 1
    assert concentration_param > 0
    assert isinstance(num_passes, int)
    assert num_passes > 0

    # set learning rate to 0; unused
    learning_rate = 0

    # dimensionality of points
    num_obs, obs_dim = observations.shape
    max_num_clusters = num_obs
    num_clusters = 1

    # centroids of clusters
    means = np.zeros(shape=(max_num_clusters, obs_dim))

    # initial cluster = first data point
    means[0, :] = observations[0, :]

    # empirical online classification labels
    cluster_assgts = np.zeros((max_num_clusters, max_num_clusters))
    cluster_assgts[0, 0] = 1

    for pass_idx in range(num_passes):
        for obs_idx in range(1, len(observations)):

            # Obtain distances between current sample and previous centroids:
            distances = np.linalg.norm(observations[obs_idx, :] - means[:num_clusters, :],
                                       axis=1)
            assert len(distances) == num_clusters

            # Create a new cluster if the smallest distance > the cutoff:
            if np.min(distances) > concentration_param:

                num_clusters += 1

                # Set centroid of new cluster = new sample
                means[num_clusters - 1, :] = observations[obs_idx, :]
                cluster_assgts[obs_idx, num_clusters - 1] = 1.

            else:
                # Assign sample to a previous cluster
                assigned_cluster = np.argmin(distances)
                cluster_assgts[obs_idx, assigned_cluster] = 1.

        for cluster_idx in range(num_clusters):
            # Obtain indices of all samples assigned to current cluster:
            indices_of_points_in_assigned_cluster = cluster_assgts[:, cluster_idx] == 1

            # Obtain the samples in the current cluster
            points_in_assigned_cluster = observations[indices_of_points_in_assigned_cluster, :]
            assert points_in_assigned_cluster.shape[0] >= 1

            # Recompute centroid after adding current sample
            means[cluster_idx, :] = np.mean(points_in_assigned_cluster,
                                            axis=0)
    cluster_assgt_posteriors_running_sum = np.cumsum(np.copy(cluster_assgts), axis=0)

    # Return the assigned classes and their centroids
    dp_means_offline_results = dict(
        cluster_assgt_posteriors=cluster_assgts,
        cluster_assgt_posteriors_running_sum=cluster_assgt_posteriors_running_sum,
        parameters=dict(means=means),
    )
    return dp_means_offline_results


# DP-GMM
def dp_gmm(observations,
           likelihood_model: str,
           learning_rate: float,
           concentration_param: float,
           max_iter: int = 8,  # same as DP-Means
           num_initializations: int = 1):
    # Variational Inference for Dirichlet Process Mixtures
    # Blei and Jordan (2006)

    assert concentration_param > 0

    num_obs, obs_dim = observations.shape
    var_dp_gmm = sklearn.mixture.BayesianGaussianMixture(
        n_components=num_obs,
        max_iter=max_iter,
        n_init=num_initializations,
        init_params='random',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=concentration_param)
    var_dp_gmm.fit(observations)
    cluster_assgt_posteriors = var_dp_gmm.predict_proba(observations)
    cluster_assgt_posteriors_running_sum = np.cumsum(cluster_assgt_posteriors,
                                                     axis=0)
    params = dict(means=var_dp_gmm.means_,
                  covs=var_dp_gmm.covariances_)

    # returns classes assigned and centroids of corresponding classes
    variational_results = dict(
        cluster_assgt_posteriors=cluster_assgt_posteriors,
        cluster_assgt_posteriors_running_sum=cluster_assgt_posteriors_running_sum,
        parameters=params,
    )
    return variational_results


def run_inference_alg(inference_alg_str,
                      observations,
                      concentration_param,
                      likelihood_model,
                      learning_rate):
    inference_alg_kwargs = dict()

    # RN-CRP
    if inference_alg_str == 'RN-CRP':
        inference_alg_fn = rn_crp

    # DP-GMM
    elif inference_alg_str.startswith('DP-GMM'):
        inference_alg_fn = dp_gmm

        substrs = inference_alg_str.split(' ')  # Parse parameters from algorithm string as needed
        num_initializations = int(substrs[2][1:])
        max_iters = int(substrs[4])

        inference_alg_kwargs['num_initializations'] = num_initializations
        inference_alg_kwargs['max_iter'] = max_iters

    # DP-Means
    elif inference_alg_str.startswith('DP-Means'):
        inference_alg_fn = dp_means

        if inference_alg_str.endswith('(offline)'):
            inference_alg_kwargs['num_passes'] = 8  # same as Kulis and Jordan

        elif inference_alg_str.endswith('(online)'):
            inference_alg_kwargs['num_passes'] = 1
        else:
            raise ValueError('Invalid DP Means')

    else:
        raise ValueError(f'Unknown inference algorithm: {inference_alg_str}')

    # Run inference algorithm
    inference_alg_results = inference_alg_fn(
        observations=observations,
        concentration_param=concentration_param,
        likelihood_model=likelihood_model,
        learning_rate=learning_rate,
        **inference_alg_kwargs)

    return inference_alg_results
