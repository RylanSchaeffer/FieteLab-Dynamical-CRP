import numpy as np
import os
import pandas as pd

from rncrp.helpers.run import download_wandb_project_runs_results
import plot_arxiv_2022

exp_dir = '08_arxiv'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/dcrp-arxiv-2022"
sweep_name = 'yooy4r3l'
sweep_dir = os.path.join(results_dir, sweep_name)
os.makedirs(sweep_dir, exist_ok=True)
sweep_results_df_path = os.path.join(sweep_dir, f'sweep={sweep_name}_results.csv')

if not os.path.isfile(sweep_results_df_path):
    sweep_results_df = download_wandb_project_runs_results(
        wandb_project_path=wandb_sweep_path,
        sweep_id=sweep_name)

    sweep_results_df.to_csv(sweep_results_df_path, index=False)

else:
    sweep_results_df = pd.read_csv(sweep_results_df_path)

print(f"Number of runs: {sweep_results_df.shape[0]} for sweep={sweep_name}")

raise NotImplementedError
# plot_arxiv_2022.plot_analyze_all_inf_algs_results(
#     all_inf_algs_results_df=sweep_results_df,
#     plot_dir=sweep_dir,
# )

print(f'Finished 08_arxiv/plot_sweep.py for sweep={sweep_name}.')
