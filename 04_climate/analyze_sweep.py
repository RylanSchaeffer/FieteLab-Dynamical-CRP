import numpy as np
import os
import pandas as pd

from rncrp.helpers.analyze import download_wandb_project_runs_results, generate_and_save_cluster_ratio_data
from rncrp.plot import plot_general
import plot_climate

exp_dir = '04_climate'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "" ## TODO: FILL IN
sweep_name = '' ## TODO: FILL IN
sweep_results_dir_path = os.path.join(results_dir, sweep_name)
os.makedirs(sweep_results_dir_path, exist_ok=True)
sweep_results_df_path = os.path.join(sweep_results_dir_path, f'sweep={sweep_name}_results.csv')


if not os.path.isfile(sweep_results_df_path):
    sweep_results_df = download_wandb_project_runs_results(
        wandb_project_path=wandb_sweep_path,
        sweep_id=sweep_name)
    sweep_results_df.to_csv(sweep_results_df_path, index=False)

else:
    sweep_results_df = pd.read_csv(sweep_results_df_path, index_col=False)

print(f"Number of runs: {sweep_results_df.shape[0]} for sweep={sweep_name}")

# Generate data for cluster ratio plots
# generate_and_save_cluster_ratio_data(all_inf_algs_results_df=sweep_results_df,
#                                                   plot_dir=sweep_dir)

# Generate plots & map visualizations
# TODO: Select periodicty and with_or_without_subclasses appropriately here (or generalize to all combinations)
plot_climate.plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=sweep_results_df,
    periodicity='annually',
    with_or_without_subclasses='with',
    plot_dir=sweep_results_dir_path,
)

print(f'Finished 04_climate/plot_sweep.py with sweep={sweep_name}.')