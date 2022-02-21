# Common plotting functions
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16  # was previously 22
sns.set_style("whitegrid")

algorithm_color_map = {
    'Dynamical-CRP': 'tab:blue',
    'DP-Means (Offline)': 'tab:orange',
    'DP-Means (Online)': 'tab:purple',
    'VI-GMM': 'tab:green',
}


def plot_sweep_results_all(sweep_results_df: pd.DataFrame,
                           plot_dir: str = 'results'):
    os.makedirs(plot_dir, exist_ok=True)

    plot_fns = [
        plot_num_clusters_by_alpha_colored_by_alg,
        plot_runtime_by_alpha_colored_by_alg,
        plot_runtime_by_dimension_colored_by_alg,
        plot_scores_by_snr_colored_by_alg,
        plot_scores_by_alpha_colored_by_alg,
        plot_scores_by_dimension_colored_by_alg,
    ]

    for plot_fn in plot_fns:
        # try:
        plot_fn(sweep_results_df=sweep_results_df,
                plot_dir=plot_dir)
        # except Exception as e:
        #     print(f'Exception: {e}')

        # Close all figure windows to not interfere with next plots
        plt.close('all')
        print(f'Plotted {str(plot_fn)}')


def plot_cluster_multiclass_classification_score_by_alpha_by_alg(sweep_results_df: pd.DataFrame,
                                                                 plot_dir: str):

    sns.lineplot(data=sweep_results_df,
                 x='alpha',
                 y='avg_finetune_acc',
                 hue='inference_alg_str',
                 palette=algorithm_color_map)
    plt.xlabel(r'$\alpha$')
    plt.legend()
    # plt.ylim(0., 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'cluster_multiclass_classification_score_by_alpha_by_alg.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

def plot_num_clusters_by_alpha_colored_by_alg(sweep_results_df: pd.DataFrame,
                                              plot_dir: str):
    sns.lineplot(data=sweep_results_df,
                 x='alpha',
                 y='Num Inferred Clusters',
                 hue='inference_alg_str',
                 palette=algorithm_color_map)

    # Can't figure out how to add another line to Seaborn, so manually adding
    # the next line of Num True Clusters.
    num_true_clusters_by_lambda = \
        sweep_results_df[['alpha', 'n_clusters']].groupby('alpha').agg({
            'n_clusters': ['mean', 'sem']
        })['n_clusters']

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

    plt.yscale('log')
    plt.xlabel(r'$\alpha$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'num_clusters_by_alpha.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_ratio_inferred_to_observed_true_clusters_vs_num_obs_by_alg(sweep_results_df: pd.DataFrame,
                                                                    plot_dir: str):
    """
    Plot the ratio (number of inferred clusters so far) / (number of true clusters seen so far)
        versus the number of observations, averaged over multiple datasets.
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    sweep_results_df['Inferred to True Cluster Ratio'] = sweep_results_df['Num Inferred Clusters']/sweep_results_df['Num True Clusters']
    sweep_results_df['obs_idx'] = (np.array(sweep_results_df.index)) % sweep_results_df['n_samples'][0] + 1

    g = sns.lineplot(data=sweep_results_df,
                     x='obs_idx',
                     y='Inferred to True Cluster Ratio',
                     hue='n_features',
                     ci='sd',
                     ax=ax,
                     legend='full',
                     palette=algorithm_color_map)

    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles[1:], labels=labels[1:]) # Remove "quantity" from legend title
    g.get_legend().set_title('Data Dimension')

    plt.xlabel('Number of Observations')
    plt.ylabel('Num Inferred Clusters / Num True Clusters')
    plt.ylim(bottom=0.)

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_ratio_inferred_to_observed_true_clusters_vs_num_obs_by_alg.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print("FIGURE SAVED TO:", plot_dir + '/plot_ratio_inferred_to_observed_true_clusters_vs_num_obs_by_alg.png')


def plot_ratio_observed_to_total_true_clusters_vs_num_obs_alg(sweep_results_df: pd.DataFrame,
                                                                 plot_dir: str):
    """
    Plot the ratio (number of observed true clusters so far) / (total number of true clusters)
        versus the number of observations, averaged over multiple datasets.
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    sweep_results_df['Observed to Total True Cluster Ratio'] = sweep_results_df['Num True Clusters']/sweep_results_df['n_clusters']
    sweep_results_df['obs_idx'] = (np.array(sweep_results_df.index)) % sweep_results_df['n_samples'][0] + 1

    g = sns.lineplot(data=sweep_results_df,
                     x='obs_idx',
                     y='Observed to Total True Cluster Ratio',
                     hue='n_features',
                     ci='sd',
                     ax=ax,
                     legend='full',
                     palette=algorithm_color_map)

    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles[1:], labels=labels[1:]) # Remove "quantity" from legend title
    g.get_legend().set_title('Data Dimension')

    plt.xlabel('Number of Observations')
    plt.ylabel('Observed Num True Clusters /\nTotal Num True Clusters')
    plt.ylim(bottom=0.)

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_ratio_observed_to_total_true_clusters_vs_num_obs_alg.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print("FIGURE SAVED TO:", plot_dir + '/plot_ratio_observed_to_total_true_clusters_vs_num_obs_alg.png')


def plot_runtime_by_alpha_colored_by_alg(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):
    sns.lineplot(data=sweep_results_df,
                 x='alpha',
                 y='Runtime',
                 hue='inference_alg_str',
                 palette=algorithm_color_map)
    plt.yscale('log')
    plt.xlabel(r'$\alpha$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'runtime_by_alpha.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_runtime_by_dimension_colored_by_alg(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):
    sns.lineplot(data=sweep_results_df,
                 x='n_features',
                 y='Runtime',
                 hue='inference_alg_str',
                 palette=algorithm_color_map,
                 err_style='bars')
    plt.yscale('log')
    plt.xlabel(r'Data Dimension')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'runtime_by_dimension.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_scores_by_snr_colored_by_alg(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):
    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:
        sns.lineplot(data=sweep_results_df,
                     x='snr',
                     y=score_column,
                     hue='inference_alg_str',
                     palette=algorithm_color_map)
        plt.xscale('log')
        plt.xlabel(r'SNR')
        plt.legend()
        # plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_snr.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_scores_by_alpha_colored_by_alg(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):
    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:
        sns.lineplot(data=sweep_results_df,
                     x='alpha',
                     y=score_column,
                     hue='inference_alg_str',
                     palette=algorithm_color_map)
        plt.xlabel(r'$\alpha$')
        plt.legend()
        # plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_alpha.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_scores_by_dimension_colored_by_alg(sweep_results_df: pd.DataFrame,
                                            plot_dir: str):
    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:
        sns.lineplot(data=sweep_results_df,
                     x='n_features',
                     y=score_column,
                     hue='inference_alg_str',
                     palette=algorithm_color_map,
                     err_style="bars",)
        plt.xlabel(r'Data Dimension')
        plt.legend()
        # plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_dimension.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
