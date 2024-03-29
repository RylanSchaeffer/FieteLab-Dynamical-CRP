import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import numpy as np
import pandas as pd
import os
import seaborn as sns
from typing import Dict

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
