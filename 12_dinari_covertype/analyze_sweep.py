import numpy as np
import os
import pandas as pd

from rncrp.helpers.analyze import download_wandb_project_runs_configs, generate_and_save_cluster_ratio_data


exp_dir = '12_dinari_covertype'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/rncrp-dinari-covertype"
sweep_names = [
    'cb4ejxw4']
sweep_dir = os.path.join(results_dir, sweep_name)
os.makedirs(sweep_dir, exist_ok=True)
sweep_results_df_path = os.path.join(sweep_dir, f'sweep={sweep_name}_results.csv')

if not os.path.isfile(sweep_results_df_path):

    sweep_results_df = download_wandb_project_runs_configs(
        wandb_project_path=wandb_sweep_path,
        sweep_ids=sweep_name)

    # Compute SNR := rho / sigma
    sweep_results_df['snr'] = sweep_results_df['centroids_prior_cov_prefactor'] \
                                              / sweep_results_df['likelihood_cov_prefactor']

    sweep_results_df.to_csv(sweep_results_df_path, index=False)

else:
    sweep_results_df = pd.read_csv(sweep_results_df_path)

print(f"Number of runs: {sweep_results_df.shape[0]} for sweep={sweep_name}")

# # Generate data for cluster ratio plots
# generate_and_save_cluster_ratio_data(all_inf_algs_results_df=sweep_results_df,
#                                                   plot_dir=sweep_dir)


print(f'Finished 02_ames_housing/analyze_sweep.py for sweep={sweep_name}.')