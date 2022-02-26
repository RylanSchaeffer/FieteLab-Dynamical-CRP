import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# common plotting functions
import rncrp.plot.plot_general


def plot_analyze_all_inf_algs_results(
        all_inf_algs_results_df: pd.DataFrame,
        num_inferred_clusters_div_num_true_clusters_by_obs_idx_df: pd.DataFrame,
        num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df: pd.DataFrame,
        num_true_clusters_div_total_num_true_clusters_by_obs_idx_df: pd.DataFrame,
        plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)

    for dynamics_str, sweep_subset_results_df in all_inf_algs_results_df.groupby('dynamics_str'):

        sweep_dynamics_str_dir = os.path.join(plot_dir, dynamics_str)
        os.makedirs(sweep_dynamics_str_dir, exist_ok=True)
        print(f'Plotting dynamics {dynamics_str}')

        if dynamics_str == 'step':
            title_str = r'$\Theta(\Delta)$'
        elif dynamics_str == 'exp':
            title_str = r'$\exp(-\Delta)$'
        elif dynamics_str == 'sinusoid':
            title_str = r'$\cos(\Delta)$'
        elif dynamics_str == 'hyperbolic':
            title_str = r'$\frac{1}{1 + \Delta}$'
        else:
            title_str = None

        ratio_dfs_and_plot_fns = [
            (num_inferred_clusters_div_num_true_clusters_by_obs_idx_df,
             rncrp.plot.plot_general.plot_num_inferred_clusters_div_num_true_clusters_by_obs_idx),
            (num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df,
             rncrp.plot.plot_general.plot_num_inferred_clusters_div_total_num_true_clusters_by_obs_idx),
            (num_true_clusters_div_total_num_true_clusters_by_obs_idx_df,
             rncrp.plot.plot_general.plot_num_true_clusters_div_total_num_true_clusters_by_obs_idx),
        ]

        for ratio_df, ratio_plot_fn in ratio_dfs_and_plot_fns:

            ratio_df = ratio_df[
                (ratio_df['inference_alg_str'] == 'Dynamical-CRP') &
                (ratio_df['dynamics_str'] == dynamics_str)]

            ratio_df = pd.melt(ratio_df,
                               id_vars=['inf_alg_results_path',
                                        'alpha',
                                        'n_features',
                                        'snr',
                                        'inference_alg_str',
                                        'dynamics_str'],
                               var_name='obs_idx',
                               value_name='cluster_ratio')

            ratio_plot_fn(
                ratio_df=ratio_df,
                plot_dir=sweep_dynamics_str_dir)

            plt.close('all')

            print(f'Plotted {ratio_plot_fn}')

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
            plot_fn(sweep_results_df=sweep_subset_results_df,
                    plot_dir=sweep_dynamics_str_dir,
                    title_str=title_str)
            # except Exception as e:
            #     print(f'Exception: {e}')

            # Close all figure windows to not interfere with next plots
            plt.close('all')
            print(f'Plotted {str(plot_fn)}')
