import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# common plotting functions
import rncrp.plot.plot_general


def plot_analyze_all_inf_algs_results(all_inf_algs_results_df: pd.DataFrame,
                                      num_inferred_clusters_div_num_true_clusters_by_obs_idx_df: pd.DataFrame,
                                      num_inferred_clusters_div_total_num_true_clusters_by_obs_idx_df: pd.DataFrame,
                                      num_true_clusters_div_total_num_true_clusters_by_obs_idx_df: pd.DataFrame,
                                      plot_dir: str):

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

            # Make sure that observation index is recognized as an integer
            ratio_df['obs_idx'] = pd.to_numeric(ratio_df['obs_idx'])

            ratio_plot_fn(
                ratio_df=ratio_df,
                plot_dir=sweep_dynamics_str_dir)

            plt.close('all')

            print(f'Plotted {ratio_plot_fn}')

        plot_fns = [
            plot_cluster_multiclass_classification_score_by_alpha_cross_kappa,
            plot_adjusted_mutual_information_by_alpha_cross_kappa,
            rncrp.plot.plot_general.plot_num_clusters_by_alpha_colored_by_alg,
            rncrp.plot.plot_general.plot_num_clusters_by_snr_colored_by_alg,
            rncrp.plot.plot_general.plot_runtime_by_alpha_colored_by_alg,
            rncrp.plot.plot_general.plot_scores_by_snr_colored_by_alg,
            rncrp.plot.plot_general.plot_scores_by_alpha_colored_by_alg,
            rncrp.plot.plot_general.plot_cluster_multiclass_classification_score_by_alpha_by_alg,
            rncrp.plot.plot_general.plot_num_inferred_clusters_div_num_true_clusters_by_obs_idx,
            rncrp.plot.plot_general.plot_num_inferred_clusters_div_total_num_true_clusters_by_obs_idx,
            rncrp.plot.plot_general.plot_num_true_clusters_div_total_num_true_clusters_by_obs_idx,
        ]

        for plot_fn in plot_fns:
            # try:
            plot_fn(sweep_results_df=sweep_subset_results_df,
                    plot_dir=sweep_dynamics_str_dir)
            # except Exception as e:
            #     print(f'Exception: {e}')

            # Close all figure windows to not interfere with next plots
            plt.close('all')
            print(f'Plotted {str(plot_fn)}')


def plot_cluster_multiclass_classification_score_by_alpha_cross_kappa(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    """
    Alpha is the concentration parameter of the Dynamical CRP.
    Kappa is the concentration parameter of the von Mises likelihood.
    """

    sns.lineplot(data=sweep_results_df[sweep_results_df['inference_alg_str'] == 'Dynamical-CRP'],
                 x='alpha',
                 y='avg_finetune_acc',
                 hue='likelihood_kappa',
                 legend='full',  # Ensures hue is treated as continuum & not binned.
                 )
    plt.xlabel(r'$\alpha$')
    plt.title('Dynamical CRP')
    plt.legend()
    # plt.ylim(0., 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'cluster_multiclass_classification_score_by_alpha_cross_kappa.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_adjusted_mutual_information_by_alpha_cross_kappa(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    """
    Alpha is the concentration parameter of the Dynamical CRP.
    Kappa is the concentration parameter of the von Mises likelihood.
    """

    sns.lineplot(data=sweep_results_df[sweep_results_df['inference_alg_str'] == 'Dynamical-CRP'],
                 x='alpha',
                 y='Adjusted Mutual Info Score',
                 hue='likelihood_kappa',
                 legend='full',  # Ensures hue is treated as continuum & not binned.
                 )
    plt.xlabel(r'$\alpha$')
    plt.title('Dynamical CRP')
    plt.legend()
    # plt.ylim(0., 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'adjusted_mutual_information_by_alpha_cross_kappa.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
