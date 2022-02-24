import joblib
import numpy as np
import os
import scipy.stats
import torch
import torchvision
from typing import Dict, Union
import tensorflow as tf
import tensorflow_probability as tfp

import rncrp.helpers.dynamics


def generate_heald_exp1a() -> Dict[str, np.ndarray]:
    """
    First experiment in Heald, Lengyel and Wolpert 2021, first part.

    Shown in Figure 1b.
    """

    phases_and_num_trials = [
        ('null', 50),
        ('exposure', 125),
        ('counter-exposure', 15),
        ('channel', 150),
    ]

    total_num_trials = sum([phase_and_num_trial[1]
                            for phase_and_num_trial in phases_and_num_trials])
    observations = np.zeros(shape=(total_num_trials, 2))
    observations_times = 1. + np.arange(total_num_trials)
    cluster_assignments = np.zeros(total_num_trials, dtype=np.int)

    phase_start_trial_idx = 0
    for phase, phase_num_trials in phases_and_num_trials:
        phase_end_trial_idx = phase_start_trial_idx + phase_num_trials
        if phase == 'null':
            pass
        elif phase == 'exposure':
            observations[phase_start_trial_idx:phase_end_trial_idx, 0] = 1.
            cluster_assignments[phase_start_trial_idx:phase_end_trial_idx] = 1
        elif phase == 'counter-exposure':
            observations[phase_start_trial_idx:phase_end_trial_idx, 1] = 1.
            cluster_assignments[phase_start_trial_idx:phase_end_trial_idx] = 2
        elif phase == 'channel':
            observations[phase_start_trial_idx:phase_end_trial_idx, :] = 1.
            cluster_assignments[phase_start_trial_idx:phase_end_trial_idx] = 3
        else:
            raise NotImplementedError

    cluster_assignments_one_hot = np.zeros((total_num_trials, total_num_trials))
    cluster_assignments_one_hot[np.arange(total_num_trials), cluster_assignments] = 1.

    heald_exp_1a_dict = dict(
        cluster_assignments=cluster_assignments,
        cluster_assignments_one_hot=cluster_assignments_one_hot,
        observations=observations,
        observations_times=observations_times,
    )

    return heald_exp_1a_dict


def generate_heald_exp1b() -> Dict[str, np.ndarray]:
    """
    Second experiment in Heald, Lengyel and Wolpert Nature 2021, second part.

    Shown in Figure 1d.
    """
    phases_and_num_trials = [
        ('null', 50),
        ('exposure', 125),
        ('counter-exposure', 15),
        ('channel', 2),
        ('exposure', 2),
        ('channel', 146),
    ]

    total_num_trials = sum([phase_and_num_trial[1]
                            for phase_and_num_trial in phases_and_num_trials])
    observations = np.zeros(shape=(total_num_trials, 2))
    observations_times = 1. + np.arange(total_num_trials)
    cluster_assignments = np.zeros(total_num_trials, dtype=np.int)

    phase_start_trial_idx = 0
    for phase, phase_num_trials in phases_and_num_trials:
        phase_end_trial_idx = phase_start_trial_idx + phase_num_trials
        if phase == 'null':
            pass
        elif phase == 'exposure':
            observations[phase_start_trial_idx:phase_end_trial_idx, 0] = 1.
            cluster_assignments[phase_start_trial_idx:phase_end_trial_idx] = 1
        elif phase == 'counter-exposure':
            observations[phase_start_trial_idx:phase_end_trial_idx, 1] = 1.
            cluster_assignments[phase_start_trial_idx:phase_end_trial_idx] = 2
        elif phase == 'channel':
            observations[phase_start_trial_idx:phase_end_trial_idx, :] = 1.
            cluster_assignments[phase_start_trial_idx:phase_end_trial_idx] = 3
        else:
            raise NotImplementedError

    cluster_assignments_one_hot = np.zeros((total_num_trials, total_num_trials))
    cluster_assignments_one_hot[np.arange(total_num_trials), cluster_assignments] = 1.

    heald_exp_1b_dict = dict(
        cluster_assignments=cluster_assignments,
        cluster_assignments_one_hot=cluster_assignments_one_hot,
        observations=observations,
        observations_times=observations_times,
    )

    return heald_exp_1b_dict


def generate_heald_exp2a() -> Dict[str, np.ndarray]:
    """
    Second experiment in Heald, Lengyel and Wolpert Nature 2021.

    Shown in Figure 2a.
    """

    raise NotImplementedError


def sample_mixture_model(num_obs: int = 100,
                         obs_dim: int = 10,
                         mixing_prior_str: str = 'rncrp',
                         mixing_distribution_params: dict = None,
                         component_prior_str: str = 'gaussian',
                         component_prior_params: dict = None,
                         **kwargs):
    # Ensure we have parameters to sample cluster assignments.
    if mixing_distribution_params is None:
        if mixing_prior_str == 'rncrp':
            mixing_distribution_params = {'alpha': 1.5,
                                          'beta': 0.,
                                          'dynamics_str': 'step'}
        elif mixing_prior_str == 'discrete':
            mixing_distribution_params = {
                'probs': np.ones(10) / 10.
            }
        else:
            raise NotImplementedError

    # Sample cluster assignments.
    if mixing_prior_str == 'rncrp':
        monte_carlo_rncrp_results = sample_dcrp(num_mc_samples=1,
                                                num_customer=num_obs,
                                                **mixing_distribution_params)
        cluster_assignments = monte_carlo_rncrp_results[
            'customer_assignments_by_customer'][0]  # take first/only MC sample
        observations_times = monte_carlo_rncrp_results['customer_times']
        dynamics_params = monte_carlo_rncrp_results['dynamics'].params
    elif mixing_prior_str == 'discrete':
        cluster_assignments = np.random.choice(
            len(mixing_distribution_params['probs']),
            p=mixing_distribution_params['probs'],
            replace=True,
            size=num_obs)
        observations_times = np.arange(len(cluster_assignments))
        dynamics_params = None
    else:
        raise NotImplementedError

    num_components = len(np.unique(cluster_assignments))

    # Ensure we have parameters to sample from component prior
    if component_prior_params is None:
        if component_prior_str == 'gaussian':
            component_prior_params = {
                'centroids_prior_cov_prefactor': 10.,
                'likelihood_cov_prefactor': 1.,
            }
        elif component_prior_str == 'vonmises-fisher':
            component_prior_params = {
                'kappa': 1.,
            }
        elif component_prior_str == 'dirichlet-multinomial':
            component_prior_params = {
                'alpha': 1.,
                'num_trials': 100,
            }
        else:
            raise NotImplementedError

    # Sample components.
    if component_prior_str == 'gaussian':
        means = np.random.multivariate_normal(
            mean=np.zeros(obs_dim),
            cov=component_prior_params['centroids_prior_cov_prefactor'] * np.eye(obs_dim),
            size=num_components)

        # all Gaussians have same covariance
        # TODO: generalize this so that arbitrary covariances can be used
        cov = component_prior_params['likelihood_cov_prefactor'] * np.eye(obs_dim)
        covs = np.repeat(
            cov[np.newaxis, :, :],
            repeats=num_components,
            axis=0)

        components = dict(component_prior_str=component_prior_str,
                          means=means,
                          covs=covs)

        observations = np.array([
            np.random.multivariate_normal(mean=means[assigned_cluster],
                                          cov=covs[assigned_cluster])
            for assigned_cluster in cluster_assignments])

    elif component_prior_str == 'vonmises_fisher':

        assert component_prior_params['likelihood_kappa'] >= 0

        tfd = tfp.distributions

        # mus dimension: num_components x obs_dim
        # each mu (row) is a unit vector
        mus = np.random.normal(loc=0., scale=1, size=(num_components, obs_dim))
        mus /= np.linalg.norm(mus, axis=1)[:, np.newaxis]

        # all vmf components have same concentration parameter kappa
        # Shape: (num_components, )
        kappas = component_prior_params['likelihood_kappa'] * np.ones(num_components)

        components = dict(component_prior_str=component_prior_str,
                          mus=mus,
                          kappas=kappas)

        # observations dimension: num_obs x obs_dim
        observations = np.array([
            tfd.VonMisesFisher(
                mean_direction=mus[assigned_cluster],
                concentration=kappas[assigned_cluster]).sample().numpy()
            for assigned_cluster in cluster_assignments])

    elif component_prior_str == 'dirichlet-multinomial':

        assert component_prior_params['alpha'] > 0
        assert component_prior_params['num_trials'] > 0

        tfd = tfp.distributions

        # Shape (num_components, )
        num_trials = component_prior_params['num_trials'] * np.ones(num_components)

        # All K classes equally likely to be drawn (here K = num_components)
        # Shape: (num_components, obs_dim)
        alphas = component_prior_params['alpha'] * np.ones((num_components, obs_dim))

        components = dict(component_prior_str=component_prior_str,
                          total_counts=num_trials,
                          concentrations=alphas)

        # Shape: num_obs x obs_dim
        observations = np.array([
            tfd.DirichletMultinomial(
                total_count=num_trials[assigned_cluster],
                concentration=alphas[assigned_cluster],
                allow_nan_stats=False).sample().numpy()
            for assigned_cluster in cluster_assignments])

    else:
        raise NotImplementedError

    cluster_assignments_one_hot = np.zeros((num_obs, num_obs))
    cluster_assignments_one_hot[np.arange(num_obs), cluster_assignments] = 1.

    mixture_model_result = dict(
        mixing_prior_str=mixing_prior_str,
        mixing_distribution_params=mixing_distribution_params,
        component_prior_str=component_prior_str,
        component_prior_params=component_prior_params,
        cluster_assignments=cluster_assignments,
        cluster_assignments_one_hot=cluster_assignments_one_hot,
        observations=observations,
        observations_times=observations_times,
        components=components,
        dynamics_params=dynamics_params,
    )

    return mixture_model_result


def sample_dcrp(num_mc_samples: int,
                num_customer: int,
                alpha: float,
                beta: float,
                dynamics_str: str,
                dynamics_params: Dict[str, float] = None
                ) -> Dict[str, Union[np.ndarray, rncrp.helpers.dynamics.Dynamics]]:
    assert alpha > 0.
    assert beta >= 0.

    dynamics = rncrp.helpers.dynamics.convert_dynamics_str_to_dynamics_obj(
        dynamics_str=dynamics_str,
        dynamics_params=dynamics_params)

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
        'dynamics': dynamics,
        'customer_times': customer_times,
        'pseudo_table_occupancies_by_customer': pseudo_table_occupancies_by_customer,
        'customer_assignments_by_customer': customer_assignments_by_customer,
        'one_hot_customer_assignments_by_customer': one_hot_customer_assignments_by_customer,
        'num_tables_by_customer': num_tables_by_customer,
    }

    return monte_carlo_rncrp_results
