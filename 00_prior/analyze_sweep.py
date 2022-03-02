import numpy as np
import os
import pandas as pd
import joblib
import scipy.stats
from itertools import product

import plot_prior

def compute_analytical_vs_monte_carlo_mse(analytical_marginals: np.ndarray,
                                          mc_marginals: np.ndarray,
                                          num_samples: list = None,
                                          alpha: float = 1.1):
    """
    Compute MSE (Frobenius norm) between analytical marginal matrix and Monte Carlo marginals matrix
    for a given run (i.e. for a particular setting of alpha).
    Dimension of each matrix: (num obs, max num features)
    """

    if num_samples is None:
        num_samples = [10, 25, 100, 250, 1000, 2500]
    one_run_mse_array = []

    for sample_num in num_samples:
        # draw sample
        rand_indices = np.random.choice(np.arange(mc_marginals.shape[0]),
                                        size=sample_num)
        mean_mc_marginals_of_rand_indices = np.mean(
            mc_marginals[rand_indices],
            axis=0)
        mse = np.square(np.linalg.norm(
            analytical_marginals - mean_mc_marginals_of_rand_indices))
        one_run_mse_array.append([sample_num, mse, alpha])
    return one_run_mse_array


# exp_dir_path = '/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/00_prior'
exp_dir_path = '00_prior'
results_dir_path = os.path.join(exp_dir_path, 'results')
os.makedirs(results_dir_path, exist_ok=True)

num_mc_samples = [5000]
alphas = [1.1, 10.78, 15.37, 30.91]
betas = [0.]
dynamics_strs = ['step', 'exp', 'sinusoid', 'hyperbolic']

for dynamics_str in dynamics_strs:
    if dynamics_str == 'step':
        dynamics_latex_str = r'$\Theta(\Delta)$'
    elif dynamics_str == 'exp':
        dynamics_latex_str = r'$\exp(-\Delta)$'
    elif dynamics_str == 'sinusoid':
        dynamics_latex_str = r'$\cos(\Delta)$'
    elif dynamics_str == 'hyperbolic':
        dynamics_latex_str = r'$\frac{1}{1 + \Delta}$'
    dynamics_latex_str = 'Time Function: ' + dynamics_latex_str

    mse_data = []
    for alpha, beta, mc_sample_num in product(alphas, betas, num_mc_samples):
        run_one_results_dir = os.path.join(results_dir_path,
                                           f'dyn={dynamics_str}_a={alpha}_b={beta}')

        crp_analytical_path = os.path.join(run_one_results_dir, 'analytical.joblib')
        analytical_dcrp_results = joblib.load(crp_analytical_path)

        monte_carlo_dcrp_path = os.path.join(run_one_results_dir, f'monte_carlo_samples={mc_sample_num}.joblib')
        monte_carlo_dcrp_results = joblib.load(filename=monte_carlo_dcrp_path)

        one_run_mse_array = compute_analytical_vs_monte_carlo_mse(
            analytical_marginals=analytical_dcrp_results['pseudo_table_occupancies_by_customer'],
            mc_marginals=monte_carlo_dcrp_results['pseudo_table_occupancies_by_customer'],
            alpha=alpha)
        mse_data.extend(one_run_mse_array)

    mse_by_customer_df = pd.DataFrame(mse_data,
                                      columns=['num_samples', 'mse', 'alpha_label'])
    plot_prior.plot_mse_analytical_vs_monte_carlo(
        mse_by_customer_df=mse_by_customer_df,
        plot_dir=results_dir_path,
        dynamics_latex_str=dynamics_latex_str,
        dynamics_str=dynamics_str)

