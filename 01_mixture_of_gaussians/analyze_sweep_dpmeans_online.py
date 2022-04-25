import numpy as np
import os
import pandas as pd

from rncrp.helpers.analyze import download_wandb_project_runs_configs, \
    generate_and_save_cluster_ratio_data
import matplotlib.pyplot as plt
import rncrp.plot.plot_general

exp_dir_path = '01_mixture_of_gaussians'
results_dir = os.path.join(exp_dir_path, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/dcrp-mixture-of-gaussians"

sweep_names = [
    '6yypeu59',  # DP-Means (Online) with lambda = 20/alpha
    'd40qja33',  # DP-Means (Online) with lambda = 20/sqrt(alpha)
    'rcrdfx0v',  # DP-Means (Online) with lambda = 20/log(alpha)
    'zf8f0tu6',  # DP-Means (Online) with lambda = 50/log(alpha)
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


def sweep_to_inference_alg_str(row: pd.Series):
    if row['Sweep'] == '6yypeu59':
        inference_alg_str = r'$\lambda = 20 / \alpha$'
    elif row['Sweep'] == 'd40qja33':
        inference_alg_str = r'$\lambda = 20 / \sqrt{\alpha}$'
    elif row['Sweep'] == 'rcrdfx0v':
        inference_alg_str = r'$\lambda = 20 / \log(\alpha)$'
    elif row['Sweep'] == 'zf8f0tu6':
        inference_alg_str = r'$\lambda = 50 / \log(\alpha)$'
    else:
        # run_group = f"{row['place_field_loss']}\n{row['place_field_values']}\n{row['place_field_normalization']}"
        raise ValueError
    return inference_alg_str


all_inf_algs_results_df['inference_alg_str'] = all_inf_algs_results_df.apply(
    sweep_to_inference_alg_str,
    axis=1)

print(f"Number of runs: {all_inf_algs_results_df.shape[0]} for sweep(s)={sweep_names_str}")

plot_fns = [
    rncrp.plot.plot_general.plot_num_clusters_by_alpha_colored_by_alg,
    rncrp.plot.plot_general.plot_runtime_by_alpha_colored_by_alg,
    rncrp.plot.plot_general.plot_runtime_by_dimension_colored_by_alg,
    rncrp.plot.plot_general.plot_scores_by_snr_colored_by_alg,
    rncrp.plot.plot_general.plot_scores_by_alpha_colored_by_alg,
    rncrp.plot.plot_general.plot_scores_by_dimension_colored_by_alg,
]

for plot_fn in plot_fns:
    # try:
    plot_fn(sweep_results_df=all_inf_algs_results_df,
            plot_dir=sweep_results_dir_path,
            hue_order=None,
            palette=None)
    # except Exception as e:
    #     print(f'Exception: {e}')

    # Close all figure windows to not interfere with next plots
    plt.close('all')
    print(f'Plotted {str(plot_fn)}')

print(f'Finished 01_mixture_of_gaussians/analyze_sweep_dpmeans_online.py for sweep={sweep_names_str}.')
