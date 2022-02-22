import joblib
import numpy as np
import os
import pandas as pd

import rncrp.data.real_nontabular
from rncrp.helpers.run import download_wandb_project_runs_results
import plot_yilun_nav_2d

exp_dir = '07_yilun_nav_2d'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/dcrp-yilun-nav2d"
sweep_name = '751j5e81'
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


# Plot W&B data
plot_yilun_nav_2d.plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=sweep_results_df,
    plot_dir=sweep_dir,
)


# Load dataset and clustering results
yilun_nav_2d_dataset = rncrp.data.real_nontabular.load_dataset_yilun_nav_2d_2022()

# We tested multiple hyperparameters for each environment. We want to plot each run.
for dynamics_str, sweep_subset_results_df in sweep_results_df.groupby('dynamics_str'):
    sweep_dynamics_str_dir = os.path.join(sweep_dir, dynamics_str)
    os.makedirs(sweep_dynamics_str_dir, exist_ok=True)
    for _, one_run_series in sweep_subset_results_df.iterrows():
        one_run_results = joblib.load(one_run_series['inf_alg_results_path'])

        # Convert e.g. '07_yilun_nav_2d/results/id=at3k1tjn.joblib' to e.g. 'id=at3k1tjn'
        joblib_file_name = one_run_series['inf_alg_results_path'].split('/')[-1][:-7]

        one_run_config = one_run_results['config']
        one_run_cluster_assignment_posteriors = one_run_results['inference_alg_results'][
            'cluster_assignment_posteriors']

        plot_yilun_nav_2d.plot_room_clusters_on_one_run(
            yilun_nav_2d_dataset=yilun_nav_2d_dataset,
            cluster_assignment_posteriors=one_run_cluster_assignment_posteriors,
            run_config=one_run_config,
            file_name=joblib_file_name,
            plot_dir=sweep_dynamics_str_dir,
            env_idx=one_run_series['repeat_idx'],
        )


print(f'Finished 07_yilun_nav_2d/plot_sweep.py for sweep={sweep_name}.')
