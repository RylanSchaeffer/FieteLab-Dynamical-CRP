import joblib
import os

import rncrp.data.real_nontabular
from rncrp.helpers.analyze import download_wandb_project_runs_configs
import plot_yilun_nav_2d

exp_dir = '07_yilun_nav_2d'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/dcrp-yilun-nav2d"
sweep_names = [
    '4reny29o',
]
sweep_names_str = ','.join(sweep_names)
print(f'Analyzing sweeps {sweep_names_str}')
sweep_results_dir_path = os.path.join(results_dir, sweep_names_str)
os.makedirs(sweep_results_dir_path, exist_ok=True)

all_inf_algs_results_df = download_wandb_project_runs_configs(
    wandb_project_path=wandb_sweep_path,
    data_dir=results_dir,
    sweep_ids=sweep_names,
    finished_only=True,
    refresh=False)

print(f"Number of runs: {all_inf_algs_results_df.shape[0]} for sweep={sweep_names_str}")


plot_yilun_nav_2d.plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=all_inf_algs_results_df,
    plot_dir=sweep_results_dir_path,
)


# Load dataset and clustering results
yilun_nav_2d_dataset = rncrp.data.real_nontabular.load_dataset_yilun_nav_2d_2022()

# We tested multiple hyperparameters for each environment. We want to plot each run.
for dynamics_str, sweep_subset_results_df in all_inf_algs_results_df.groupby('dynamics_str'):
    sweep_dynamics_str_dir = os.path.join(sweep_results_dir_path, dynamics_str)
    os.makedirs(sweep_dynamics_str_dir, exist_ok=True)
    for _, one_run_series in sweep_subset_results_df.iterrows():
        try:
            one_run_results = joblib.load(one_run_series['inf_alg_results_path'])
        except TypeError:
            # Somehow, the W&B path is NaN. This throws a
            # TypeError: integer argument expected, got float.
            # Just skip these.
            continue

        # Convert e.g. '07_yilun_nav_2d/results/id=at3k1tjn.joblib' to e.g. 'id=at3k1tjn'
        joblib_file_name = one_run_series['inf_alg_results_path'].split('/')[-1][:-7]

        one_run_config = one_run_results['config']
        one_run_cluster_assignment_posteriors = one_run_results['inference_alg_results'][
            'cluster_assignment_posteriors']

        plot_yilun_nav_2d.plot_room_clusters_on_one_run(
            yilun_nav_2d_dataset=yilun_nav_2d_dataset,
            cluster_assignment_posteriors=one_run_cluster_assignment_posteriors,
            run_config=one_run_config,
            plot_dir=sweep_dynamics_str_dir,
            env_idx=one_run_series['repeat_idx'],
        )


print(f'Finished 07_yilun_nav_2d/plot_sweep.py for sweep={sweep_names}.')
