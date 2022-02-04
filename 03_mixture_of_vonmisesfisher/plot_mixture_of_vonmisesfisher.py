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
        rncrp.plot.plot_general.plot_sweep_results_all(
            sweep_results_df=sweep_subset_results_df,
            plot_dir=sweep_dynamics_str_dir)
