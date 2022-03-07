import numpy as np
import os
import pandas as pd

import plot_mixture_of_gaussians
from rncrp.helpers.analyze import download_wandb_project_runs_results, \
    generate_and_save_cluster_ratio_data

exp_dir_path = '01_mixture_of_gaussians'
results_dir = os.path.join(exp_dir_path, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/dcrp-mixture-of-gaussians"
sweep_name = '4jfwfeas'
sweep_results_dir_path = os.path.join(results_dir, sweep_name)
os.makedirs(sweep_results_dir_path, exist_ok=True)
sweep_results_df_path = os.path.join(sweep_results_dir_path, f'sweep={sweep_name}_results.csv')

if not os.path.isfile(sweep_results_df_path):

    sweep_results_df = download_wandb_project_runs_results(
        wandb_project_path=wandb_sweep_path,
        sweep_id=sweep_name)

    # Compute SNR := rho / sigma
    sweep_results_df['snr'] = np.sqrt(
        sweep_results_df['centroids_prior_cov_prefactor'] \
        / sweep_results_df['likelihood_cov_prefactor'])

    sweep_results_df.to_csv(sweep_results_df_path, index=False)

else:
    sweep_results_df = pd.read_csv(sweep_results_df_path)

print(f"Number of runs: {sweep_results_df.shape[0]} for sweep={sweep_name}")

cluster_ratio_dfs_results = generate_and_save_cluster_ratio_data(
    all_inf_algs_results_df=sweep_results_df,
    sweep_results_dir_path=sweep_results_dir_path)

# Join cluster ratio dataframes with sweep hyperparameters
#     Dataframes will have columns:
#         1) 'inf_alg_results_path'
#         2) 'alpha'
#         3) 'n_features'
#         4) 'snr'
#         5) 'dynamics_str',
#         6) 'inference_alg_str'
#         7-on) '0', '1', ..., 'max number of observations.'
num_inferred_clusters_div_num_true_clusters_by_obs_idx_df = pd.merge(
    left=sweep_results_df[['inf_alg_results_path', 'alpha', 'n_features', 'snr', 'dynamics_str', 'inference_alg_str']],
    right=cluster_ratio_dfs_results['num_inferred_clusters_div_num_true_clusters_by_obs_idx_df'],
    how='inner',
    on='inf_alg_results_path')
num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df = pd.merge(
    left=sweep_results_df[['inf_alg_results_path', 'alpha', 'n_features', 'snr', 'dynamics_str', 'inference_alg_str']],
    right=cluster_ratio_dfs_results['num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df'],
    how='inner',
    on='inf_alg_results_path')
num_true_clusters_div_total_num_true_clusters_by_obs_idx_df = pd.merge(
    left=sweep_results_df[['inf_alg_results_path', 'alpha', 'n_features', 'snr', 'dynamics_str', 'inference_alg_str']],
    right=cluster_ratio_dfs_results['num_true_clusters_div_total_num_true_clusters_by_obs_idx_df'],
    how='inner',
    on='inf_alg_results_path')

# Generate all plots
plot_mixture_of_gaussians.plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=sweep_results_df,
    num_inferred_clusters_div_num_true_clusters_by_obs_idx_df=num_inferred_clusters_div_num_true_clusters_by_obs_idx_df,
    num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df=num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df,
    num_true_clusters_div_total_num_true_clusters_by_obs_idx_df=num_true_clusters_div_total_num_true_clusters_by_obs_idx_df,
    plot_dir=sweep_results_dir_path,
)

print(f'Finished 01_mixture_of_gaussians/analyze_sweep.py for sweep={sweep_name}.')
