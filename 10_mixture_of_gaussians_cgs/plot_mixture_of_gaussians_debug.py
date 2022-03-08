from itertools import product
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

    plot_num_clusters_by_alpha_split_by_hyperparameter_choices(
        sweep_results_df=all_inf_algs_results_df,
        plot_dir=plot_dir,
    )

    plot_scores_by_alpha_split_by_hyperparameter_choices(
        sweep_results_df=all_inf_algs_results_df,
        plot_dir=plot_dir,
    )
    #
    # for dynamics_str, sweep_subset_results_df in all_inf_algs_results_df.groupby('dynamics_str'):
    #
    #     sweep_dynamics_str_dir = os.path.join(plot_dir, dynamics_str)
    #     os.makedirs(sweep_dynamics_str_dir, exist_ok=True)
    #     print(f'Plotting dynamics {dynamics_str}')
    #
    #     if dynamics_str == 'step':
    #         title_str = r'$\Theta(\Delta)$'
    #     elif dynamics_str == 'exp':
    #         title_str = r'$\exp(-\Delta)$'
    #     elif dynamics_str == 'sinusoid':
    #         title_str = r'$\cos(\Delta)$'
    #     elif dynamics_str == 'hyperbolic':
    #         title_str = r'$\frac{1}{1 + \Delta}$'
    #     else:
    #         title_str = None
    #
    #     ratio_dfs_and_plot_fns = [
    #         (num_inferred_clusters_div_num_true_clusters_by_obs_idx_df,
    #          rncrp.plot.plot_general.plot_num_inferred_clusters_div_num_true_clusters_by_obs_idx),
    #         (num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df,
    #          rncrp.plot.plot_general.plot_num_inferred_clusters_div_total_num_true_clusters_by_obs_idx),
    #         (num_true_clusters_div_total_num_true_clusters_by_obs_idx_df,
    #          rncrp.plot.plot_general.plot_num_true_clusters_div_total_num_true_clusters_by_obs_idx),
    #     ]
    #
    #     for ratio_df, ratio_plot_fn in ratio_dfs_and_plot_fns:
    #
    #         ratio_df = ratio_df[
    #             (ratio_df['inference_alg_str'] == 'Dynamical-CRP') &
    #             (ratio_df['dynamics_str'] == dynamics_str)]
    #
    #         # If sweep doesn't contain dynamical CRP, ratio_df will be empty.
    #         if len(ratio_df) == 0:
    #             continue
    #
    #         ratio_df = pd.melt(ratio_df,
    #                            id_vars=['inf_alg_results_path',
    #                                     'alpha',
    #                                     'n_features',
    #                                     'snr',
    #                                     'inference_alg_str',
    #                                     'dynamics_str'],
    #                            var_name='obs_idx',
    #                            value_name='cluster_ratio')
    #
    #         # Make sure that observation index is recognized as an integer
    #         ratio_df['obs_idx'] = pd.to_numeric(ratio_df['obs_idx'])
    #
    #         ratio_plot_fn(
    #             ratio_df=ratio_df,
    #             plot_dir=sweep_dynamics_str_dir)
    #
    #         plt.close('all')
    #
    #         print(f'Plotted {ratio_plot_fn}')
    #
    #     plot_fns = [
    #         rncrp.plot.plot_general.plot_num_clusters_by_alpha_colored_by_alg,
    #         rncrp.plot.plot_general.plot_runtime_by_alpha_colored_by_alg,
    #         rncrp.plot.plot_general.plot_runtime_by_dimension_colored_by_alg,
    #         rncrp.plot.plot_general.plot_scores_by_snr_colored_by_alg,
    #         rncrp.plot.plot_general.plot_scores_by_alpha_colored_by_alg,
    #         rncrp.plot.plot_general.plot_scores_by_dimension_colored_by_alg,
    #     ]
    #
    #     for plot_fn in plot_fns:
    #         # try:
    #         plot_fn(sweep_results_df=sweep_subset_results_df,
    #                 plot_dir=sweep_dynamics_str_dir,
    #                 title_str=title_str)
    #         # except Exception as e:
    #         #     print(f'Exception: {e}')
    #
    #         # Close all figure windows to not interfere with next plots
    #         plt.close('all')
    #         print(f'Plotted {str(plot_fn)}')


def plot_num_clusters_by_alpha_split_by_hyperparameter_choices(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    for hue in product(('zero', 'observation'), ('DP', 'variational'), (True, False), (True, False)):
        plt.close()

        # Manually make figure bigger to handle external legend
        plt.figure(figsize=(9, 4))

        sweep_results_df_subset = sweep_results_df[
            (sweep_results_df['vi_param_initialization'] == hue[0])
            & (sweep_results_df['which_prior_prob'] == hue[1])
            & (sweep_results_df['update_new_cluster_parameters'] == hue[2])
            & (sweep_results_df['robbins_monro_cavi_updates'] == hue[3])]

        # Can't figure out how to add another line to Seaborn, so manually adding
        # the next line of Num True Clusters.
        num_true_clusters_by_lambda = \
            sweep_results_df_subset[['alpha', 'n_clusters']].groupby('alpha').agg({
                'n_clusters': ['mean', 'sem']
            })['n_clusters']

        sns.lineplot(data=sweep_results_df_subset,
                     x='alpha',
                     y='Num Inferred Clusters',
                     hue='inference_alg_str',
                     )

        means = num_true_clusters_by_lambda['mean'].values
        sems = num_true_clusters_by_lambda['sem'].values
        plt.plot(
            num_true_clusters_by_lambda.index.values,
            means,
            label='Num True Clusters',
            color='k',
        )
        plt.fill_between(
            x=num_true_clusters_by_lambda.index.values,
            y1=means - sems,
            y2=means + sems,
            alpha=0.3,
            linewidth=0,
            color='k')

        plt.ylim(10., 100.)
        plt.yscale('log')
        plt.xlabel(r'$\alpha$')
        plt.title(hue)

        # Move legend outside of plot
        # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
        plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        plt.tight_layout()

        plt.savefig(os.path.join(plot_dir,
                                 f'num_clusters_by_alpha_init={hue[0]}_prior={hue[1]}_updatenew={hue[2]}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_scores_by_alpha_split_by_hyperparameter_choices(sweep_results_df: pd.DataFrame,
                                                         plot_dir: str,
                                                         title_str: str = None):

    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for hue in product(('zero', 'observation'), ('DP', 'variational'), (True, False), (True, False)):

        sweep_results_df_subset = sweep_results_df[
            (sweep_results_df['vi_param_initialization'] == hue[0])
            & (sweep_results_df['which_prior_prob'] == hue[1])
            & (sweep_results_df['update_new_cluster_parameters'] == hue[2])
            & (sweep_results_df['robbins_monro_cavi_updates'] == hue[3])]

        for score_column in scores_columns:

            plt.close()

            # Manually make figure bigger to handle external legend
            plt.figure(figsize=(9, 4))

            sns.lineplot(data=sweep_results_df_subset,
                         x='alpha',
                         y=score_column,
                         hue='inference_alg_str')
            plt.xlabel(r'$\alpha$')
            # plt.legend()

            # Move legend outside of plot
            # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
            plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

            if title_str is not None:
                plt.title(title_str)

            plt.ylim(0.35, 0.9)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir,
                                     f'comparison_score={score_column}_by_alpha_init={hue[0]}_prior={hue[1]}_updatenew={hue[2]}.png'),
                        bbox_inches='tight',
                        dpi=300)
            # plt.show()
            plt.close()
