import numpy as np
import os
import pandas as pd
import joblib

from rncrp.helpers.run import download_wandb_project_runs_results
import plot_mixture_of_gaussians


def generate_and_save_data_for_cluster_ratio_plotting(all_inf_algs_results_df: pd.DataFrame,
                                                      plot_dir: str):

    inferred_to_true_data_paths_array = []
    observed_to_total_true_data_paths_array = []

    # count = 0
    for dynamics_str, sweep_subset_results_df in all_inf_algs_results_df.groupby('dynamics_str'):
        sweep_dynamics_str_dir = os.path.join(plot_dir, dynamics_str)
        os.makedirs(sweep_dynamics_str_dir, exist_ok=True)
        for param_tuple, df_by_n_features in sweep_subset_results_df.groupby(['alpha',
                                                                              'likelihood_cov_prefactor',
                                                                              'centroids_prior_cov_prefactor',
                                                                              'inference_alg_str']):
            # if count==2:
            #     break

            if param_tuple[-1] == 'Dynamical-CRP':
                param_tuple_dir = '_'.join(str(x) for x in param_tuple)
                cluster_ratio_plot_dir = os.path.join(sweep_dynamics_str_dir, param_tuple_dir)
                os.makedirs(cluster_ratio_plot_dir, exist_ok=True)

                all_inferred_to_true_data_array = []
                all_observed_to_total_true_data_array = []

                for joblib_and_n_features, filtered_df in df_by_n_features.groupby(['inf_alg_results_path',
                                                                                              'n_features']):

                    data_dim = joblib_and_n_features[1]

                    joblib_subpath = joblib_and_n_features[0]
                    joblib_file = joblib.load('/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/'+joblib_subpath)

                    # Obtain number of inferred clusters
                    cluster_assignment_posteriors = joblib_file['inference_alg_results']['cluster_assignment_posteriors']
                    # cluster_assignment_priors = joblib_file['inference_alg_results']['cluster_assignment_priors']
                    inferred_cluster_assignments = cluster_assignment_posteriors.argmax(axis=1)
                    inferred_clusters_so_far = np.array([len(np.unique(inferred_cluster_assignments[:i+1])) for i in range(len(inferred_cluster_assignments))])

                    # Obtain number of observed true clusters
                    true_cluster_assignments = joblib_file['mixture_model_results']['cluster_assignments']
                    true_clusters_seen_so_far = np.array([len(np.unique(true_cluster_assignments[:i+1])) for i in range(len(true_cluster_assignments))])

                    # Generate data for plotting
                    seq_length = cluster_assignment_posteriors.shape[0]
                    obs_indices = 1 + np.arange(seq_length)
                    inferred_to_true_data_df = pd.DataFrame.from_dict({
                        'obs_idx': obs_indices,
                        'data_dim': np.array([data_dim]*seq_length),
                        'cluster_ratio': inferred_clusters_so_far / true_clusters_seen_so_far,
                    })
                    observed_to_total_true_data_df = pd.DataFrame.from_dict({
                        'obs_idx': obs_indices,
                        'data_dim': np.array([data_dim]*seq_length),
                        'cluster_ratio': true_clusters_seen_so_far / max(true_clusters_seen_so_far),
                    })

                    all_inferred_to_true_data_array.append(inferred_to_true_data_df)
                    all_observed_to_total_true_data_array.append(observed_to_total_true_data_df)

                concatenated_inferred_to_true_df = pd.concat(all_inferred_to_true_data_array, ignore_index=True)
                concatenated_observed_to_total_true_df = pd.concat(all_observed_to_total_true_data_array, ignore_index=True)

                concatenated_inferred_to_true_df.to_pickle(cluster_ratio_plot_dir+'/concatenated_inferred_to_true_df.pkl')
                concatenated_observed_to_total_true_df.to_pickle(cluster_ratio_plot_dir+'/concatenated_observed_to_total_true_df.pkl')

                inferred_to_true_data_paths_array.append(cluster_ratio_plot_dir+'/concatenated_inferred_to_true_df.pkl')
                observed_to_total_true_data_paths_array.append(cluster_ratio_plot_dir+'/concatenated_observed_to_total_true_df.pkl')
                # count += 1

    np.save('inferred_to_true_data_paths_array.npy',np.array(inferred_to_true_data_paths_array))
    np.save('observed_to_total_true_data_paths_array.npy', np.array(observed_to_total_true_data_paths_array))


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

generate_and_save_data_for_cluster_ratio_plotting(all_inf_algs_results_df=sweep_results_df,
                                                  plot_dir=sweep_dir)

plot_mixture_of_gaussians.plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=sweep_results_df,
    plot_dir=sweep_dir,
)

print(f'Finished 01_mixture_of_gaussians/analyze_sweep.py for sweep={sweep_name}.')
