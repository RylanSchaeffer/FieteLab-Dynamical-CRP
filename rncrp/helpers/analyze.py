import os
import pandas as pd
import numpy as np
import joblib
import wandb


def download_wandb_project_runs_results(wandb_project_path: str,
                                        sweep_id: str = None,
                                        ) -> pd.DataFrame:
    # Download sweep results
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    if sweep_id is None:
        runs = api.runs(path=wandb_project_path)
    else:
        runs = api.runs(path=wandb_project_path,
                        filters={"Sweep": sweep_id})

    sweep_results_list = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        summary.update(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        summary.update({'State': run.state,
                        'Sweep': run.sweep.id if run.sweep is not None else None})
        # .name is the human-readable name of the run.
        summary.update({'run_name': run.name})
        sweep_results_list.append(summary)

    sweep_results_df = pd.DataFrame(sweep_results_list)

    # Keep only finished runs
    finished_runs = sweep_results_df['State'] == 'finished'
    print(f"% of successfully finished runs: {finished_runs.mean()}")
    sweep_results_df = sweep_results_df[finished_runs]

    if sweep_id is not None:
        sweep_results_df = sweep_results_df[sweep_results_df['Sweep'] == sweep_id]

    # Ensure we aren't working with a slice.
    sweep_results_df = sweep_results_df.copy()

    return sweep_results_df


def generate_and_save_data_for_cluster_ratio_plotting(all_inf_algs_results_df: pd.DataFrame,
                                                      plot_dir: str):
    # count = 0
    for dynamics_str, sweep_subset_results_df in all_inf_algs_results_df.groupby('dynamics_str'):
        sweep_dynamics_str_dir = os.path.join(plot_dir, dynamics_str)
        os.makedirs(sweep_dynamics_str_dir, exist_ok=True)

        inferred_to_observed_true_data_paths_array = []
        inferred_to_total_true_data_paths_array = []
        observed_true_to_total_true_data_paths_array = []

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

                all_inferred_to_observed_true_data_array = []
                all_inferred_to_total_true_data_array = []
                all_observed_true_to_total_true_data_array = []

                for joblib_and_n_features, filtered_df in df_by_n_features.groupby(['inf_alg_results_path',
                                                                                    'n_features']):
                    data_dim = joblib_and_n_features[1]

                    joblib_subpath = joblib_and_n_features[0]
                    joblib_file = joblib.load(
                        '/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/' + joblib_subpath)

                    # Obtain number of inferred clusters
                    cluster_assignment_posteriors = joblib_file['inference_alg_results'][
                        'cluster_assignment_posteriors']
                    # cluster_assignment_priors = joblib_file['inference_alg_results']['cluster_assignment_priors']
                    inferred_cluster_assignments = cluster_assignment_posteriors.argmax(axis=1)
                    inferred_clusters_so_far = np.array([len(np.unique(inferred_cluster_assignments[:i + 1])) for i in
                                                         range(len(inferred_cluster_assignments))])

                    # Obtain numbers of observed and total true clusters
                    true_cluster_assignments = joblib_file['mixture_model_results']['cluster_assignments']
                    true_clusters_seen_so_far = np.array([len(np.unique(true_cluster_assignments[:i + 1])) for i in
                                                          range(len(true_cluster_assignments))])

                    # Generate data for plotting
                    seq_length = cluster_assignment_posteriors.shape[0]
                    obs_indices = 1 + np.arange(seq_length)
                    inferred_to_observed_true_data_df = pd.DataFrame.from_dict({
                        'obs_idx': obs_indices,
                        'data_dim': np.array([data_dim] * seq_length),
                        'cluster_ratio': inferred_clusters_so_far / true_clusters_seen_so_far,
                    })
                    inferred_to_total_true_data_df = pd.DataFrame.from_dict({
                        'obs_idx': obs_indices,
                        'data_dim': np.array([data_dim] * seq_length),
                        'cluster_ratio': inferred_clusters_so_far / max(true_clusters_seen_so_far),
                    })
                    observed_true_to_total_true_data_df = pd.DataFrame.from_dict({
                        'obs_idx': obs_indices,
                        'data_dim': np.array([data_dim] * seq_length),
                        'cluster_ratio': true_clusters_seen_so_far / max(true_clusters_seen_so_far),
                    })

                    all_inferred_to_observed_true_data_array.append(inferred_to_observed_true_data_df)
                    all_inferred_to_total_true_data_array.append(inferred_to_total_true_data_df)
                    all_observed_true_to_total_true_data_array.append(observed_true_to_total_true_data_df)

                # Concatenate data to average over in the plots
                concatenated_inferred_to_observed_true_df = pd.concat(all_inferred_to_observed_true_data_array,
                                                                      ignore_index=True)
                concatenated_inferred_to_total_true_df = pd.concat(all_inferred_to_total_true_data_array,
                                                                   ignore_index=True)
                concatenated_observed_true_to_total_true_df = pd.concat(all_observed_true_to_total_true_data_array,
                                                                        ignore_index=True)

                # Save each dataframe
                concatenated_inferred_to_observed_true_df.to_pickle(
                    cluster_ratio_plot_dir + '/concatenated_inferred_to_observed_true_df.pkl')
                concatenated_inferred_to_total_true_df.to_pickle(
                    cluster_ratio_plot_dir + '/concatenated_inferred_to_total_true_df.pkl')
                concatenated_observed_true_to_total_true_df.to_pickle(
                    cluster_ratio_plot_dir + '/concatenated_observed_true_to_total_true_df.pkl')

                # Store paths to the concatenated dataframes
                inferred_to_observed_true_data_paths_array.append(
                    cluster_ratio_plot_dir + '/concatenated_inferred_to_observed_true_df.pkl')
                inferred_to_total_true_data_paths_array.append(
                    cluster_ratio_plot_dir + '/concatenated_inferred_to_total_true_df.pkl')
                observed_true_to_total_true_data_paths_array.append(
                    cluster_ratio_plot_dir + '/concatenated_observed_true_to_total_true_df.pkl')
                # count += 1

        # TODO: Check that specified directory works
        np.save(os.path.join(sweep_dynamics_str_dir, 'inferred_to_observed_true_data_paths_array.npy'),
                np.array(inferred_to_observed_true_data_paths_array))
        np.save(os.path.join(sweep_dynamics_str_dir, 'inferred_to_true_true_data_paths_array.npy'),
                np.array(inferred_to_total_true_data_paths_array))
        np.save(os.path.join(sweep_dynamics_str_dir, 'observed_true_to_total_true_data_paths_array.npy'),
                np.array(observed_true_to_total_true_data_paths_array))
