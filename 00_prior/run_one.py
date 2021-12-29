"""
Compare the analytical DCRP marginal distribution against
Monte-Carlo estimates of the DCRP marginal distribution.

Example usage:

00_prior/run_one.py --results_dir_path=00_prior/results \
 --num_customer=50 --num_mc_sample=2000 \
 --alpha=2.51 --beta=3.7 --dynamics_str=exp
"""

import argparse
import logging
import joblib
import numpy as np
import os
import scipy.stats
from sympy.functions.combinatorial.numbers import stirling
from typing import Dict


import plot
import utils.data
import utils.helpers


def run_one(args: argparse.Namespace):

    run_one_results_dir = setup(args=args)

    sample_dcrp_results = sample_dcrp_and_save(
        num_customer=args.num_customer,
        alpha=args.alpha,
        beta=args.beta,
        run_one_results_dir=run_one_results_dir,
        num_mc_sample=args.num_mc_sample,
        dynamics_str=args.dynamics_str,
        time_sampling_str=args.time_sampling_str)

    analytical_dcrp_results = compute_analytical_dcrp_and_save(
        num_customer=args.num_customer,
        alpha=args.alpha,
        beta=args.beta,
        run_one_results_dir=run_one_results_dir,
        dynamics_str=args.dynamics_str,
        customer_times=sample_dcrp_results['customer_times'])

    plot.plot_customer_assignments_analytical_vs_monte_carlo(
        sampled_customer_assignments_by_customer=sample_dcrp_results['one_hot_customer_assignments_by_customer'],
        analytical_customer_assignments_by_customer=analytical_dcrp_results['customer_assignment_probs_by_customer'],
        alpha=args.alpha,
        beta=args.beta,
        plot_dir=run_one_results_dir)

    plot.plot_num_tables_analytical_vs_monte_carlo(
        sampled_num_tables_by_customer=sample_dcrp_results['num_tables_by_customer'],
        analytical_num_tables_by_customer=analytical_dcrp_results['num_table_probs_by_customer'],
        alpha=args.alpha,
        beta=args.beta,
        plot_dir=run_one_results_dir)


def chinese_table_restaurant_distribution(t, k, alpha):
    if k > t:
        prob = 0.
    else:
        prob = scipy.special.gamma(alpha)
        prob *= stirling(n=t, k=k, kind=1, signed=False)
        prob /= scipy.special.gamma(alpha + t)
        prob *= np.power(alpha, k)
    return prob


def compute_analytical_dcrp(num_customer: int,
                            alpha: float,
                            beta: float,
                            dynamics_str: str,
                            customer_times: np.ndarray) -> Dict[str, np.ndarray]:

    """

    The analytics depend on the exact times that the data were observed. Consequently,
    to match the Monte Carlo samples and the analytical results, we need access to the
    data times. customer_times has shape (num customers, )

    :param num_customer:
    :param alpha:
    :param beta:
    :param dynamics_str:
    :param customer_times:
    :return:
    """

    assert customer_times.shape == (num_customer, )

    dynamics = utils.helpers.convert_dynamics_str_to_dynamics_obj(
        dynamics_str=dynamics_str)

    # Create arrays to store all information.
    # To make Python indexing match the mathematical notation, we'll use 1-based
    # indexing and then cut off the extra row and column at the end.
    customer_assignment_probs_by_customer = np.zeros(
        shape=(num_customer + 1, num_customer + 1))
    pseudo_table_occupancies_by_customer = np.zeros_like(
        customer_assignment_probs_by_customer)
    num_table_probs_by_customer = np.zeros_like(
        customer_assignment_probs_by_customer)
    num_table_poisson_rate_by_customer = np.zeros(
        shape=(num_customer + 1,))

    if dynamics_str == 'perfectintegrator':
        # Can do additional error checking for the CRP
        analytical_crt_probs_by_customer = np.zeros(
            shape=(1 + num_customer, 1 + num_customer))
        analytical_crt_probs_by_customer[1, 1] = 1.

    # First datum requires specific setup
    customer_assignment_probs_by_customer[1, 1] = 1.
    pseudo_table_occupancies_by_customer[1, 1] = 1.
    num_table_probs_by_customer[1, 1] = 1.
    num_table_poisson_rate_by_customer[1] = 1.

    dynamics.initialize_state(
        customer_assignment_probs=customer_assignment_probs_by_customer[1, :],
        time=customer_times[0])

    # All remaining datum use same sequence of steps
    for cstmr_idx in range(2, num_customer + 1):

        # Note: customer times is 0-indexed, not 1-indexed.
        state = dynamics.run_dynamics(
            time_start=customer_times[cstmr_idx - 2],
            time_end=customer_times[cstmr_idx - 1])
        current_pseudo_table_occupancies = state['N'].copy()

        # Record current pseudo table occupancies
        pseudo_table_occupancies_by_customer[cstmr_idx, :] = current_pseudo_table_occupancies.copy()

        # Add alpha, normalize and add that distribution to running statistics
        weighted_alpha = alpha * num_table_probs_by_customer[cstmr_idx - 1, :]
        current_pseudo_table_occupancies[1:cstmr_idx+1] += weighted_alpha[:cstmr_idx]  # right shift

        if dynamics_str == 'perfectintegrator':
            # check for correctness, if possible
            assert np.allclose(np.sum(current_pseudo_table_occupancies), alpha + cstmr_idx - 1)

        normalization_const = np.sum(current_pseudo_table_occupancies)
        current_customer_assignment_probs = np.divide(
            current_pseudo_table_occupancies,
            normalization_const)

        # record values
        customer_assignment_probs_by_customer[cstmr_idx, :] = current_customer_assignment_probs.copy()
        pseudo_table_occupancies_by_customer[cstmr_idx, :] += current_customer_assignment_probs.copy()

        # update dynamics state
        # Note: customer times is 0-indexed, not 1-indexed.
        dynamics.update_state(
            customer_assignment_probs=current_customer_assignment_probs,
            time=customer_times[cstmr_idx - 1],)

        # Update distribution over number of clusters in complexity O(t)
        new_table_prob = alpha / normalization_const
        num_table_probs_by_customer[cstmr_idx, :cstmr_idx] += \
            (1 - new_table_prob) * num_table_probs_by_customer[cstmr_idx-1, :cstmr_idx]
        num_table_probs_by_customer[cstmr_idx, 1:1+cstmr_idx] += \
            new_table_prob * num_table_probs_by_customer[cstmr_idx-1, :cstmr_idx]

        # Check that we have a proper distribution over the number of tables
        assert np.allclose(np.sum(num_table_probs_by_customer[cstmr_idx, :]), 1.)

        if dynamics_str == 'perfectintegrator':
            # check for correctness, if possible
            for k_idx in range(1 + cstmr_idx):
                analytical_crt_probs_by_customer[cstmr_idx, k_idx] = chinese_table_restaurant_distribution(
                    t=cstmr_idx,
                    k=k_idx,
                    alpha=alpha)
            assert np.allclose(analytical_crt_probs_by_customer, num_table_probs_by_customer)

    # Cutoff extra row and columns we introduced at the beginning.
    customer_assignment_probs_by_customer = customer_assignment_probs_by_customer[1:, 1:]
    pseudo_table_occupancies_by_customer = pseudo_table_occupancies_by_customer[1:, 1:]
    num_table_probs_by_customer = num_table_probs_by_customer[1:, 1:]

    analytical_dcrp_results = {
        'customer_times': customer_times,
        'pseudo_table_occupancies_by_customer': pseudo_table_occupancies_by_customer,
        'customer_assignment_probs_by_customer': customer_assignment_probs_by_customer,
        'num_table_probs_by_customer': num_table_probs_by_customer,
    }

    return analytical_dcrp_results


def compute_analytical_dcrp_and_save(num_customer: int,
                                     alpha: float,
                                     beta: float,
                                     dynamics_str: str,
                                     customer_times: np.ndarray,
                                     run_one_results_dir: str) -> Dict[str, np.ndarray]:

    crp_analytical_path = os.path.join(
        run_one_results_dir,
        'analytical.joblib')

    # if not os.path.isfile(crp_analytical_path):
    analytical_dcrp_results = compute_analytical_dcrp(
        num_customer=num_customer,
        alpha=alpha,
        beta=beta,
        dynamics_str=dynamics_str,
        customer_times=customer_times)
    logging.info(f'Computed analytical results for {crp_analytical_path}')
    joblib.dump(filename=crp_analytical_path,
                value=analytical_dcrp_results)

    # this gives weird error: joblib ValueError: EOF: reading array data, expected 262144 bytes got 225056
    # analytical_dcrp_results = joblib.load(crp_analytical_path)
    assert analytical_dcrp_results['customer_times'].shape \
           == (num_customer, )
    assert analytical_dcrp_results['pseudo_table_occupancies_by_customer'].shape \
           == (num_customer, num_customer)
    assert analytical_dcrp_results['customer_assignment_probs_by_customer'].shape \
           == (num_customer, num_customer)
    assert analytical_dcrp_results['num_table_probs_by_customer'].shape \
           == (num_customer, num_customer)

    logging.info(f'Loaded analytical results for {crp_analytical_path}')
    return analytical_dcrp_results


def sample_dcrp_and_save(num_customer: int,
                         alpha: float,
                         beta: float,
                         num_mc_sample: int,
                         dynamics_str: str,
                         time_sampling_str: str,
                         run_one_results_dir: str) -> Dict[str, np.ndarray]:

    crp_samples_path = os.path.join(
        run_one_results_dir,
        f'monte_carlo_samples={num_mc_sample}.joblib')

    # if not os.path.isfile(crp_samples_path):

    sample_dcrp_results = utils.data.sample_dcrp(
        num_mc_sample=num_mc_sample,
        num_customer=num_customer,
        alpha=alpha,
        beta=beta,
        dynamics_str=dynamics_str,
        time_sampling_str=time_sampling_str)
    logging.info(f'Generated samples for {crp_samples_path}')
    joblib.dump(filename=crp_samples_path,
                value=sample_dcrp_results)

    # sample_dcrp_results = joblib.load(filename=crp_samples_path)
    assert sample_dcrp_results['customer_times'].shape \
           == (num_customer, )
    assert sample_dcrp_results['pseudo_table_occupancies_by_customer'].shape \
           == (num_mc_sample, num_customer, num_customer)
    assert sample_dcrp_results['customer_assignments_by_customer'].shape \
           == (num_mc_sample, num_customer)
    assert sample_dcrp_results['one_hot_customer_assignments_by_customer'].shape \
           == (num_mc_sample, num_customer, num_customer)
    assert sample_dcrp_results['num_tables_by_customer'].shape \
           == (num_mc_sample, num_customer, num_customer)

    logging.info(f'Loaded samples for {crp_samples_path}')
    return sample_dcrp_results


# def construct_analytical_crt(T,
#                              alphas):
#     table_nums = 1 + np.arange(T)
#     table_distributions_by_alpha_by_T = {}
#     for alpha in alphas:
#         table_distributions_by_alpha_by_T[alpha] = {}
#         for t in table_nums:
#             result = np.zeros(shape=T)
#             for repeat_idx in np.arange(1, 1 + t):
#                 result[repeat_idx - 1] = chinese_table_restaurant_distribution(
#                     t=t,
#                     k=repeat_idx,
#                     alpha=alpha)
#             table_distributions_by_alpha_by_T[alpha][t] = result
#     return table_distributions_by_alpha_by_T


def setup(args: argparse.Namespace):
    run_one_results_dir = os.path.join(
        args.results_dir_path,
        f'dyn={args.dynamics_str}_a={args.alpha}_b={args.beta}_time={args.time_sampling_str}')
    os.makedirs(run_one_results_dir, exist_ok=True)
    np.random.seed(args.seed)
    return run_one_results_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--results_dir_path', type=str,
                        help='Path to write plots and other results to.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Pseudo-random seed for NumPy/PyTorch.')
    parser.add_argument('--num_customer', type=int,
                        help='Number of customers per Monte Carlo sample.')
    parser.add_argument('--num_mc_sample', type=int,
                        help='Number of Monte Carlo samples from conditional.')
    parser.add_argument('--alpha', type=float,
                        help='DCRP alpha parameter.')
    parser.add_argument('--beta', type=float,
                        help='DCRP beta parameter.')
    parser.add_argument('--dynamics_str', type=str,
                        choices=utils.dynamics.dynamics_strs,
                        help='Choice of Temporal decay function.')
    parser.add_argument('--time_sampling_str', type=str, default='identity',
                        choices=utils.helpers.time_sampling_strs,
                        help='How index times should be sampled.')
    args = parser.parse_args()
    run_one(args)
    logging.info('Finished.')
