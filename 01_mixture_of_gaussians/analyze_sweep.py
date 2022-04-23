import numpy as np
import os
import pandas as pd

import plot_mixture_of_gaussians
from rncrp.helpers.analyze import download_wandb_project_runs_configs, \
    generate_and_save_cluster_ratio_data

exp_dir_path = '01_mixture_of_gaussians'
results_dir = os.path.join(exp_dir_path, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/dcrp-mixture-of-gaussians"
sweep_names = [
    'w3ytp57f',  # D-CRP
    'wlrc2asb',  # R-CRP
    'lovo5rgf',  # K-Means (Offline)
    'ro1zrkea',  # K-Means (Online)
    'i1h6gz8e',  # VI-GMM
    'bj7ihoq8',  # DP-Means (Offline)
    '6yypeu59',  # DP-Means (Online)
    'a56b0b8r',  # Collapsed Gibbs Sampler
]
sweep_names_str = ','.join(sweep_names)
print(f'Analyzing sweeps {sweep_names_str}')
sweep_results_dir_path = os.path.join(results_dir, sweep_names_str)
os.makedirs(sweep_results_dir_path, exist_ok=True)
sweep_results_df_path = os.path.join(sweep_results_dir_path, f'sweeps={sweep_names_str}_results.csv')


all_inf_algs_results_df = download_wandb_project_runs_configs(
    wandb_project_path=wandb_sweep_path,
    data_dir=results_dir,
    sweep_ids=sweep_names,
    finished_only=True,
    refresh=False)


# Compute SNR := rho / sigma
all_inf_algs_results_df['snr'] = np.sqrt(
    all_inf_algs_results_df['centroids_prior_cov_prefactor'] \
    / all_inf_algs_results_df['likelihood_cov_prefactor'])


print(f"Number of runs: {all_inf_algs_results_df.shape[0]} for sweep(s)={sweep_names_str}")

cluster_ratio_dfs_results = generate_and_save_cluster_ratio_data(
    all_inf_algs_results_df=all_inf_algs_results_df,
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
    left=all_inf_algs_results_df[['inf_alg_results_path', 'alpha', 'n_features', 'snr', 'dynamics_str', 'inference_alg_str']],
    right=cluster_ratio_dfs_results['num_inferred_clusters_div_num_true_clusters_by_obs_idx_df'],
    how='inner',
    on='inf_alg_results_path')
num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df = pd.merge(
    left=all_inf_algs_results_df[['inf_alg_results_path', 'alpha', 'n_features', 'snr', 'dynamics_str', 'inference_alg_str']],
    right=cluster_ratio_dfs_results['num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df'],
    how='inner',
    on='inf_alg_results_path')
num_true_clusters_div_total_num_true_clusters_by_obs_idx_df = pd.merge(
    left=all_inf_algs_results_df[['inf_alg_results_path', 'alpha', 'n_features', 'snr', 'dynamics_str', 'inference_alg_str']],
    right=cluster_ratio_dfs_results['num_true_clusters_div_total_num_true_clusters_by_obs_idx_df'],
    how='inner',
    on='inf_alg_results_path')

# Generate all plots
plot_mixture_of_gaussians.plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=all_inf_algs_results_df,
    num_inferred_clusters_div_num_true_clusters_by_obs_idx_df=num_inferred_clusters_div_num_true_clusters_by_obs_idx_df,
    num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df=num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df,
    num_true_clusters_div_total_num_true_clusters_by_obs_idx_df=num_true_clusters_div_total_num_true_clusters_by_obs_idx_df,
    plot_dir=sweep_results_dir_path,
)

print(f'Finished 01_mixture_of_gaussians/analyze_sweep.py for sweep={sweep_names_str}.')
