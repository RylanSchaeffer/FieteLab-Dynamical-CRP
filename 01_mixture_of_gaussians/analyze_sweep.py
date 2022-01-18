import numpy as np
import os
import pandas as pd

from rncrp.helpers.run import download_wandb_project_runs_results
from rncrp.plot import plot_sweep_results_all

exp_dir = '01_mixture_of_gaussians'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/rncrp-mixture-of-gaussians"
sweep_name = 'usd703j8'
sweep_dir = os.path.join(results_dir, sweep_name)
os.makedirs(sweep_dir, exist_ok=True)
sweep_results_df_path = os.path.join(sweep_dir, f'sweep={sweep_name}_results.csv')

if not os.path.isfile(sweep_results_df_path):

    sweep_results_df = download_wandb_project_runs_results(
        wandb_project_path=wandb_sweep_path,
        sweep_name=sweep_name)

    # Compute SNR := rho / sigma
    sweep_results_df['cov_prefactor_ratio'] = sweep_results_df['centroids_prior_cov_prefactor'] \
                                              / sweep_results_df['likelihood_cov_prefactor']

    sweep_results_df.to_csv(sweep_results_df_path, index=False)
else:
    sweep_results_df = pd.read_csv(sweep_results_df_path, index_col=False)

plot_sweep_results_all(sweep_results_df=sweep_results_df,
                       plot_dir=sweep_dir)

print(f'Finished 01_mixture_of_gaussians/plot_sweep.py with sweep={sweep_name}.')
