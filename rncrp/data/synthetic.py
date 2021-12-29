import joblib
import numpy as np
import os
import scipy.stats
import torch
import torchvision
from typing import Dict, Union

import rncrp.helpers.dynamics


# Generate mixture of gaussians
def generate_gaussian_params_from_gaussian_prior(num_gaussians: int = 3,
                                                 gaussian_dim: int = 2,
                                                 feature_prior_cov_scaling: float = 3.,
                                                 gaussian_cov_scaling: float = 0.3):
    # Sample Gaussian means from prior N(0, rho * I)
    means = np.random.multivariate_normal(
        mean=np.zeros(gaussian_dim),
        cov=feature_prior_cov_scaling * np.eye(gaussian_dim),
        size=num_gaussians)

    cov = gaussian_cov_scaling * np.eye(gaussian_dim)
    covs = np.repeat(cov[np.newaxis, :, :],
                     repeats=num_gaussians,
                     axis=0)

    mixture_of_gaussians = dict(means=means, covs=covs)
    return mixture_of_gaussians


# Sample from mixture
def sample_from_mixture_of_gaussians(seq_len: int = 100,
                                              num_gaussians: int = None,
                                              gaussian_params: dict = {}):

    assert num_gaussians is not None
    assigned_table_seq = np.random.choice(np.arange(num_gaussians),
                                          replace=True,
                                          size=seq_len)

    mixture_of_gaussians = generate_mixture_of_gaussians(num_gaussians=num_gaussians, **gaussian_params)

    gaussian_samples_seq = np.array([
        np.random.multivariate_normal(mean=mixture_of_gaussians['means'][assigned_table],
                                      cov=mixture_of_gaussians['covs'][assigned_table])
        for assigned_table in assigned_table_seq])

    assigned_table_seq_one_hot = np.zeros((seq_len, seq_len))
    assigned_table_seq_one_hot[np.arange(seq_len), assigned_table_seq] = 1.

    result = dict(
        mixture_of_gaussians=mixture_of_gaussians,
        assigned_table_seq=assigned_table_seq,
        assigned_table_seq_one_hot=assigned_table_seq_one_hot,
        gaussian_samples_seq=gaussian_samples_seq
    )

    return result


def sample_ibp(num_mc_sample: int,
               num_customer: int,
               alpha: float,
               beta: float) -> Dict[str, np.ndarray]:
    assert alpha > 0.
    assert beta > 0.

    # preallocate results
    # use 10 * expected number of dishes as heuristic
    max_dishes = 10 * int(alpha * beta * np.sum(1 / (1 + np.arange(num_customer))))
    sampled_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)
    cum_sampled_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)
    num_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)

    for smpl_idx in range(num_mc_sample):
        current_num_dishes = 0
        for cstmr_idx in range(1, num_customer + 1):
            # sample existing dishes
            prob_new_customer_sampling_dish = cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 2, :] / \
                                              (beta + cstmr_idx - 1)
            existing_dishes_for_new_customer = np.random.binomial(
                n=1,
                p=prob_new_customer_sampling_dish[np.newaxis, :])[0]
            sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1,
            :] = existing_dishes_for_new_customer  # .astype(np.int)

            # sample number of new dishes for new customer
            # subtract 1 from to cstmr_idx because of 1-based iterating
            num_new_dishes = np.random.poisson(alpha * beta / (beta + cstmr_idx - 1))
            start_dish_idx = current_num_dishes
            end_dish_idx = current_num_dishes + num_new_dishes
            sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, start_dish_idx:end_dish_idx] = 1

            # increment current num dishes
            current_num_dishes += num_new_dishes
            num_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, current_num_dishes] = 1

            # increment running sums
            cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, :] = np.add(
                cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 2, :],
                sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, :])

    sample_ibp_results = {
        'cum_sampled_dishes_by_customer_idx': cum_sampled_dishes_by_customer_idx,
        'sampled_dishes_by_customer_idx': sampled_dishes_by_customer_idx,
        'num_dishes_by_customer_idx': num_dishes_by_customer_idx,
    }

    # import matplotlib.pyplot as plt
    # mc_avg = np.mean(sample_ibp_results['sampled_dishes_by_customer_idx'], axis=0)
    # plt.imshow(mc_avg)
    # plt.show()

    return sample_ibp_results


def sample_from_linear_gaussian(num_obs: int = 100,
                                indicator_sampling_str: str = 'categorical',
                                indicator_sampling_params: Dict[str, float] = None,
                                feature_prior_params: Dict[str, float] = None,
                                likelihood_params: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    """
    Draw sample from Binary Linear-Gaussian model.

    :return:
        sampled_indicators: NumPy array with shape (seq_len,) of (integer) sampled classes
        linear_gaussian_samples_seq: NumPy array with shape (seq_len, obs_dim) of
                                binary linear-Gaussian samples
    """

    if feature_prior_params is None:
        feature_prior_params = {}

    if likelihood_params is None:
        likelihood_params = {'sigma_x': 1e-1}

    # Otherwise, use categorical or IBP to sample number of features
    if indicator_sampling_str not in {'categorical', 'IBP', 'GriffithsGhahramani'}:
        raise ValueError(f'Impermissible class sampling value: {indicator_sampling_str}')

    # Unique case of generating Griffiths & Ghahramani data
    if indicator_sampling_str == 'GriffithsGhahramani':
        sampled_data_result = sample_from_griffiths_ghahramani_2005(
            num_obs=num_obs,
            gaussian_likelihood_params=likelihood_params)
        sampled_data_result['indicator_sampling_str'] = indicator_sampling_str

        indicator_sampling_descr_str = '{}_probs=[{}]'.format(
            indicator_sampling_str,
            ','.join([str(i) for i in sampled_data_result['indicator_sampling_params']['probs']]))
        sampled_data_result['indicator_sampling_descr_str'] = indicator_sampling_descr_str
        return sampled_data_result

    if indicator_sampling_str is None:
        indicator_sampling_params = dict()

    if indicator_sampling_str == 'categorical':

        # if probabilities per cluster aren't specified, go with uniform probabilities
        if 'probs' not in indicator_sampling_params:
            indicator_sampling_params['probs'] = np.ones(5) / 5

        indicator_sampling_descr_str = '{}_probs={}'.format(
            indicator_sampling_str,
            list(indicator_sampling_params['probs']))
        indicator_sampling_descr_str = indicator_sampling_descr_str.replace(' ', '')

    elif indicator_sampling_str == 'IBP':
        if 'alpha' not in indicator_sampling_params:
            indicator_sampling_params['alpha'] = 3.98
        if 'beta' not in indicator_sampling_params:
            indicator_sampling_params['beta'] = 4.97
        indicator_sampling_descr_str = '{}_a={}_b={}'.format(
            indicator_sampling_str,
            indicator_sampling_params['alpha'],
            indicator_sampling_params['beta'])

    else:
        raise NotImplementedError

    if indicator_sampling_str == 'categorical':
        num_gaussians = indicator_sampling_params['probs'].shape[0]
        sampled_indicators = np.random.binomial(
            n=1,
            p=indicator_sampling_params['probs'][np.newaxis, :],
            size=(num_obs, num_gaussians))
    elif indicator_sampling_str == 'IBP':
        sampled_indicators = sample_ibp(
            num_mc_sample=1,
            num_customer=num_obs,
            alpha=indicator_sampling_params['alpha'],
            beta=indicator_sampling_params['beta'])['sampled_dishes_by_customer_idx'][0, :, :]
        num_gaussians = np.argwhere(np.sum(sampled_indicators, axis=0) == 0.)[0, 0]
        sampled_indicators = sampled_indicators[:, :num_gaussians]
    else:
        raise ValueError(f'Impermissible class sampling: {indicator_sampling_str}')

    gaussian_params = generate_gaussian_params_from_gaussian_prior(
        num_gaussians=num_gaussians,
        **feature_prior_params)

    features = gaussian_params['means']
    obs_dim = features.shape[1]
    obs_means = np.matmul(sampled_indicators, features)
    obs_cov = np.square(likelihood_params['sigma_x']) * np.eye(obs_dim)
    observations = np.array([
        np.random.multivariate_normal(
            mean=obs_means[obs_idx],
            cov=obs_cov)
        for obs_idx in range(num_obs)])

    sampled_data_result = dict(
        gaussian_params=gaussian_params,
        sampled_indicators=sampled_indicators,
        observations=observations,
        features=features,
        indicator_sampling_str=indicator_sampling_str,
        indicator_sampling_params=indicator_sampling_params,
        indicator_sampling_descr_str=indicator_sampling_descr_str,
        feature_prior_params=feature_prior_params,
        likelihood_params=likelihood_params,
    )

    return sampled_data_result


def sample_rncrp(num_mc_samples: int,
                 num_customer: int,
                 alpha: float,
                 beta: float,
                 dynamics_str: str,
                 ) -> Dict[str, np.ndarray]:

    assert alpha > 0.
    assert beta >= 0.

    dynamics = rncrp.helpers.dynamics.convert_dynamics_str_to_dynamics_obj(
        dynamics_str=dynamics_str)

    # time_sampling_fn = utils.helpers.convert_time_sampling_str_to_time_sampling_fn(
    #     time_sampling_str=time_sampling_str)

    def time_sampling_fn(num_customer: int) -> np.ndarray:
        return 1. + np.arange(num_customer)

    customer_times = time_sampling_fn(num_customer=num_customer)

    pseudo_table_occupancies_by_customer = np.zeros(
        shape=(num_mc_samples, num_customer, num_customer))
    one_hot_customer_assignments_by_customer = np.zeros(
        shape=(num_mc_samples, num_customer, num_customer))
    customer_assignments_by_customer = np.zeros(
        shape=(num_mc_samples, num_customer,),
        dtype=np.int)
    num_tables_by_customer = np.zeros(
        shape=(num_mc_samples, num_customer, num_customer))

    # the first customer always goes at the first table
    pseudo_table_occupancies_by_customer[:, 0, 0] = 1
    one_hot_customer_assignments_by_customer[:, 0, 0] = 1.
    num_tables_by_customer[:, 0, 0] = 1.

    for mc_sample_idx in range(num_mc_samples):
        new_table_idx = 1
        dynamics.initialize_state(
            customer_assignment_probs=one_hot_customer_assignments_by_customer[mc_sample_idx, 0, :],
            time=customer_times[0])
        for cstmr_idx in range(1, num_customer):
            state = dynamics.run_dynamics(
                time_start=customer_times[cstmr_idx - 1],
                time_end=customer_times[cstmr_idx])
            current_pseudo_table_occupancies = state['N']
            pseudo_table_occupancies_by_customer[mc_sample_idx, cstmr_idx, :] = current_pseudo_table_occupancies.copy()

            # Add alpha, normalize and sample from that distribution.
            current_pseudo_table_occupancies = current_pseudo_table_occupancies.copy()
            current_pseudo_table_occupancies[new_table_idx] = alpha
            probs = current_pseudo_table_occupancies / np.sum(current_pseudo_table_occupancies)
            customer_assignment = np.random.choice(np.arange(new_table_idx + 1),
                                                   p=probs[:new_table_idx + 1])
            assert customer_assignment < cstmr_idx + 1

            # store sampled customer
            one_hot_customer_assignments_by_customer[mc_sample_idx, cstmr_idx, customer_assignment] = 1.
            new_table_idx = max(new_table_idx, customer_assignment + 1)
            num_tables_by_customer[mc_sample_idx, cstmr_idx, new_table_idx - 1] = 1.
            customer_assignments_by_customer[mc_sample_idx, cstmr_idx] = customer_assignment

            # Increment psuedo-table occupancies
            state = dynamics.update_state(
                customer_assignment_probs=one_hot_customer_assignments_by_customer[mc_sample_idx, cstmr_idx, :],
                time=customer_times[cstmr_idx])
            pseudo_table_occupancies_by_customer[mc_sample_idx, cstmr_idx, :] = state['N'].copy()

    monte_carlo_rncrp_results = {
        'customer_times': customer_times,
        'pseudo_table_occupancies_by_customer': pseudo_table_occupancies_by_customer,
        'customer_assignments_by_customer': customer_assignments_by_customer,
        'one_hot_customer_assignments_by_customer': one_hot_customer_assignments_by_customer,
        'num_tables_by_customer': num_tables_by_customer,
    }

    return monte_carlo_rncrp_results
