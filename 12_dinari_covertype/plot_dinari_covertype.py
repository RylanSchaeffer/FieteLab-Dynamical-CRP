import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# common plotting functions
import rncrp.plot.plot_general


def plot_analyze_all_inf_algs_results(all_inf_algs_results_df: pd.DataFrame,
                                      plot_dir: str):

    for dynamics_str, sweep_subset_results_df in all_inf_algs_results_df.groupby('dynamics_str'):

        sweep_dynamics_str_dir = os.path.join(plot_dir, dynamics_str)
        os.makedirs(sweep_dynamics_str_dir, exist_ok=True)
        print(f'Plotting dynamics {dynamics_str}')

        os.makedirs(plot_dir, exist_ok=True)

        plot_fns = [
            rncrp.plot.plot_general.plot_num_clusters_by_alpha_colored_by_alg,
            rncrp.plot.plot_general.plot_runtime_by_alpha_colored_by_alg,
            rncrp.plot.plot_general.plot_scores_by_snr_colored_by_alg,
            rncrp.plot.plot_general.plot_scores_by_alpha_colored_by_alg,
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


def plot_dinari_covertype_labels(labels: np.ndarray,
                                 plot_dir: str,
                                 title_str: str = None):

    plt.close()
    index = 1 + np.arange(len(labels))
    sns.scatterplot(index, labels, s=1)
    plt.xlabel('Obs Index')
    plt.ylabel('Forest Cover Type (Class Label)')

    if title_str is not None:
        plt.title(title_str)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'dinari_covertype_labels.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
