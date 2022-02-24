import os
import pandas as pd

from rncrp.helpers.analyze import download_wandb_project_runs_results, generate_and_save_data_for_cluster_ratio_plotting
import plot_mixture_of_gaussians

exp_dir = '/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/01_mixture_of_gaussians'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/dcrp-mixture-of-gaussians"
sweep_name = '9kplnw7y'
sweep_dir = os.path.join(results_dir, sweep_name)
os.makedirs(sweep_dir, exist_ok=True)
sweep_results_df_path = os.path.join(sweep_dir, f'sweep={sweep_name}_results.csv')

if not os.path.isfile(sweep_results_df_path):
    sweep_results_df = download_wandb_project_runs_results(
        wandb_project_path=wandb_sweep_path,
        sweep_id=sweep_name)

    # Compute SNR := rho / sigma
    sweep_results_df['snr'] = sweep_results_df['centroids_prior_cov_prefactor'] \
                              / sweep_results_df['likelihood_cov_prefactor']

    sweep_results_df.to_csv(sweep_results_df_path, index=False)

else:
    sweep_results_df = pd.read_csv(sweep_results_df_path)

print(f"Number of runs: {sweep_results_df.shape[0]} for sweep={sweep_name}")

# TODO: Add this to analyze_sweep.py for any setting needing cluster ratio plots
generate_and_save_data_for_cluster_ratio_plotting(all_inf_algs_results_df=sweep_results_df,
                                                  plot_dir=sweep_dir)

plot_mixture_of_gaussians.plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=sweep_results_df,
    plot_dir=sweep_dir,
)

print(f'Finished 01_mixture_of_gaussians/analyze_sweep.py for sweep={sweep_name}.')
