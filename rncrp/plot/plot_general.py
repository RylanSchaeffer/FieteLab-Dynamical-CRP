# Common plotting functions
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List
import joblib

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16  # was previously 22
sns.set_style("whitegrid")

algorithm_color_map = {
    'DP-Means (Offline)': 'tab:orange',
    'DP-Means (Online)': 'tab:purple',
    'Dynamical-CRP': 'tab:blue',
    'Dynamical-CRP (Cutoff=1e-2)': 'tab:red',
    'Dynamical-CRP (Cutoff=1e-4)': 'tab:brown',
    'K-Means (Offline)': 'tab:pink',
    'K-Means (Online)': 'tab:gray',
    'Recursive-CRP': 'tab:olive',
    'VI-GMM': 'tab:green',
    'CollapsedGibbsSampler': 'tab:cyan',
}


def plot_cluster_multiclass_classification_score_by_alpha_by_alg(sweep_results_df: pd.DataFrame,
                                                                 plot_dir: str,
                                                                 title_str: str = None):
    # Manually make figure bigger to handle external legend
    plt.figure(figsize=(9, 4))

    sns.lineplot(data=sweep_results_df,
                 x='alpha',
                 y='avg_finetune_acc',
                 hue='inference_alg_str',
                 palette=algorithm_color_map)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Finetune Accuracy')
    plt.legend()

    if title_str is not None:
        plt.title(title_str)

    # plt.ylim(0., 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'cluster_multiclass_classification_score_by_alpha_by_alg.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_cluster_assignments_inferred_vs_true(cluster_assignment_posteriors: np.ndarray,
                                              true_cluster_assignments_one_hot: np.ndarray,
                                              plot_dir: str,
                                              plot_title: str = None,
                                              default_num_tables_to_plot: int = 100):
    """
    Plot true cluster assignments (Y = observation index, X = cluster assignment) (left)
    and inferred cluster assignments (Y = observation index, X = cluster assignment) (right).

    Args:
        cluster_assignment_posteriors: Shape (num obs, max num clusters)
        true_cluster_assignments_one_hot: Shape (num obs, max num clusters)
    """

    num_tables_to_plot = min([default_num_tables_to_plot,
                              true_cluster_assignments_one_hot.shape[1],
                              cluster_assignment_posteriors.shape[1]])

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=(8, 4))

    if plot_title is not None:
        fig.suptitle(plot_title)

    ax_idx = 0
    # plot prior table assignments
    ax = axes[ax_idx]
    true_cluster_assignments_subset = true_cluster_assignments_one_hot[
                                      :num_tables_to_plot, :num_tables_to_plot]
    sns.heatmap(true_cluster_assignments_one_hot[:num_tables_to_plot, :num_tables_to_plot],
                ax=ax,
                cmap='Blues',
                # xticklabels=1 + np.arange(num_tables_to_plot),
                mask=np.isnan(true_cluster_assignments_subset),
                vmin=0.,
                vmax=1.)
    ax.set_title(r'True Clusters')
    ax.set_ylabel('Observation Index')
    ax.set_xlabel('Cluster Index')

    # plot posterior table assignments
    ax_idx += 1
    ax = axes[ax_idx]
    cluster_assignment_posteriors_subset = cluster_assignment_posteriors[
                                           :num_tables_to_plot, :num_tables_to_plot]
    sns.heatmap(cluster_assignment_posteriors_subset,
                ax=ax,
                cmap='Blues',
                # xticklabels=1 + np.arange(num_tables_to_plot),
                mask=np.isnan(cluster_assignment_posteriors_subset),
                vmin=0.,
                vmax=1.)
    ax.set_title(r'Inferred Clusters')
    ax.set_xlabel('Cluster Index')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'cluster_assignments_inferred_vs_true.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

    plt.savefig('path_to_save.png')
    plt.close()


def plot_cluster_coassignments_inferred_vs_true(cluster_assignment_posteriors: np.ndarray,
                                                true_cluster_assignments: np.ndarray,
                                                plot_dir: str,
                                                plot_title: str = None,
                                                default_num_obs_to_plot: int = 100):
    """
    Plot observation-by-observation matrices of inner products between cluster posteriors.
    True cluster assignments on the left, inferred on the right.

    We permute the observations to group same cluster IDs.

    Args:
        cluster_assignment_posteriors: Shape (num obs, max num clusters)
        true_cluster_assignments: Shape (num obs,)
    """

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=(8, 4))

    if plot_title is not None:
        fig.suptitle(plot_title)

    num_obs_to_plot = min([default_num_obs_to_plot,
                           true_cluster_assignments.shape[0]])

    ax_idx = 0
    # plot prior table assignments
    ax = axes[ax_idx]
    true_cluster_assignments_subset = true_cluster_assignments[:num_obs_to_plot]

    true_cluster_assignments_subset_one_hot = np.zeros((num_obs_to_plot, num_obs_to_plot))
    true_cluster_assignments_subset_one_hot[
        np.arange(num_obs_to_plot), true_cluster_assignments_subset] = 1.

    true_pairwise_similarities_subset = np.matmul(
        true_cluster_assignments_subset_one_hot,
        true_cluster_assignments_subset_one_hot.T)

    # Compute indices to reshuffle based on true cluster assignment.
    shuffle_indices = np.argsort(true_cluster_assignments_subset)

    # Shuffle to group together similar clusters.
    true_pairwise_similarities_subset = true_pairwise_similarities_subset[
        shuffle_indices, :][:, shuffle_indices]

    sns.heatmap(true_pairwise_similarities_subset,
                ax=ax,
                cmap='Blues',
                mask=np.isnan(true_pairwise_similarities_subset),
                vmin=0.,
                vmax=1.)
    ax.set_title(r'True Similarities')
    ax.set_ylabel('Sorted Obs Index')
    ax.set_xlabel('Sorted Obs Index')

    # plot posterior table assignments
    ax_idx += 1
    ax = axes[ax_idx]
    cluster_assignment_posteriors_subset = cluster_assignment_posteriors[
                                           :num_obs_to_plot, :num_obs_to_plot]
    inferred_pairwise_similarities_subset = np.matmul(
        cluster_assignment_posteriors_subset,
        cluster_assignment_posteriors_subset.T)

    # plt.close()
    # shuffle_indices = np.argsort(true_cluster_assignments)
    # sns.heatmap(np.matmul(cluster_assignment_posteriors[true_cluster_assignments],
    #                       cluster_assignment_posteriors[true_cluster_assignments].T),
    #             cmap='Blues')
    # plt.show()

    # Shuffle to group together similar clusters.
    inferred_pairwise_similarities_subset = inferred_pairwise_similarities_subset[
        shuffle_indices, :][:, shuffle_indices]

    sns.heatmap(inferred_pairwise_similarities_subset,
                ax=ax,
                cmap='Blues',
                # xticklabels=1 + np.arange(num_tables_to_plot),
                mask=np.isnan(inferred_pairwise_similarities_subset),
                vmin=0.,
                vmax=1.)
    ax.set_title(r'Inferred Similarities')
    ax.set_ylabel('Sorted Obs Index')
    ax.set_xlabel('Sorted Obs Index')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'cluster_coassignments_inferred_vs_true.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()


def plot_num_clusters_by_alpha_colored_by_alg(sweep_results_df: pd.DataFrame,
                                              plot_dir: str,
                                              title_str: str = None):

    # Manually make figure bigger to handle external legend
    plt.figure(figsize=(9, 4))

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

    # Move legend outside of plot
    # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
    plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    if title_str is not None:
        plt.title(title_str)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'num_clusters_by_alpha.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_num_clusters_by_snr_colored_by_alg(sweep_results_df: pd.DataFrame,
                                            plot_dir: str,
                                            title_str: str = None):

    # Manually make figure bigger to handle external legend
    plt.figure(figsize=(9, 4))

    sns.lineplot(data=sweep_results_df,
                 x='snr',
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
    plt.xlabel(r'SNR')

    if title_str is not None:
        plt.title(title_str)

    # plt.legend()

    # Move legend outside of plot
    # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
    plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'num_clusters_by_snr.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_num_inferred_clusters_div_num_true_clusters_by_obs_idx(ratio_df: pd.DataFrame,
                                                                plot_dir: str,
                                                                title_str: str = None):
    """
    Plot the ratio of (number of inferred clusters so far) / (number of true clusters seen so far)
        versus the number of observations, averaged over multiple datasets.
    """

    # (alpha, likelihood_cov_prefactor, centroids_prior_cov_prefactor)
    g = sns.lineplot(data=ratio_df,
                     x='obs_idx',
                     y='cluster_ratio',
                     hue='n_features',
                     ci='sd',
                     legend='full')
    # palette=algorithm_color_map)

    # Rylan 2022/02/26: I don't these two lines actually remove quantity from legend title.
    # handles, labels = g.get_legend_handles_labels()
    # g.legend(handles=handles[1:], labels=labels[1:])  # Remove "quantity" from legend title

    g.get_legend().set_title('Data Dimension')

    plt.xlabel('Num. of Observations')
    plt.ylabel('Num. Inferred Clusters /\nNum. True Clusters Seen')
    # plt.ylim(bottom=0.)
    plt.yscale('log')

    if title_str is not None:
        plt.title(title_str)

    plt.grid()
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, 'num_inferred_clusters_div_num_true_clusters_by_obs_idx_df.png'),
        bbox_inches='tight',
        dpi=300)
    # plt.show()
    plt.close()


def plot_num_inferred_clusters_div_total_num_true_clusters_by_obs_idx(ratio_df: pd.DataFrame,
                                                                      plot_dir: str,
                                                                      title_str: str = None):
    """
    Plot the ratio of (number of inferred clusters so far) / (total number of true clusters)
        versus the number of observations, averaged over multiple datasets.
    """

    # (alpha, likelihood_cov_prefactor, centroids_prior_cov_prefactor)
    g = sns.lineplot(data=ratio_df,
                     x='obs_idx',
                     y='cluster_ratio',
                     hue='n_features',
                     ci='sd',
                     legend='full')
    # palette=algorithm_color_map)

    # Rylan 2022/02/26: I don't these two lines actually remove quantity from legend title.
    # handles, labels = g.get_legend_handles_labels()
    # g.legend(handles=handles[1:], labels=labels[1:])  # Remove "quantity" from legend title
    g.get_legend().set_title('Data Dimension')

    plt.xlabel('Num. of Observations')
    plt.ylabel('Num. Inferred Clusters /\nTotal Num. True Clusters')
    # plt.ylim(bottom=0.)
    plt.yscale('log')

    if title_str is not None:
        plt.title(title_str)

    plt.grid()
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, 'num_inferred_clusters_div_total_num_true_clusters_by_obs_idx.png'),
        bbox_inches='tight',
        dpi=300)
    # plt.show()
    plt.close()


def plot_num_true_clusters_div_total_num_true_clusters_by_obs_idx(ratio_df: pd.DataFrame,
                                                                  plot_dir: str,
                                                                  title_str: str = None):
    """
    Plot the ratio of (number of true clusters so far) / (number of total true clusters)
        versus the number of observations, averaged over multiple datasets.
    """

    g = sns.lineplot(data=ratio_df,
                     x='obs_idx',
                     y='cluster_ratio',
                     hue='n_features',
                     ci='sd',
                     legend='full')
    # palette=algorithm_color_map)

    # Rylan 2022/02/26: I don't these two lines actually remove quantity from legend title.
    # handles, labels = g.get_legend_handles_labels()
    # g.legend(handles=handles[1:], labels=labels[1:])  # Remove "quantity" from legend title

    g.get_legend().set_title('Data Dimension')

    plt.xlabel('Num. of Observations')
    plt.ylabel('Num. True Clusters /\nTotal Num. True Clusters')
    plt.ylim(bottom=0.)

    if title_str is not None:
        plt.title(title_str)

    plt.grid()
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, 'num_true_clusters_div_total_num_true_clusters_by_obs_idx.png'),
        bbox_inches='tight',
        dpi=300)
    # plt.show()
    plt.close()


def plot_runtime_by_alpha_colored_by_alg(sweep_results_df: pd.DataFrame,
                                         plot_dir: str,
                                         title_str: str = None):
    # Manually make figure bigger to handle external legend
    plt.figure(figsize=(9, 4))

    sns.lineplot(data=sweep_results_df,
                 x='alpha',
                 y='Runtime',
                 hue='inference_alg_str',
                 palette=algorithm_color_map)
    plt.yscale('log')
    plt.xlabel(r'$\alpha$')

    if title_str is not None:
        plt.title(title_str)

    # Move legend outside of plot
    # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
    plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'runtime_by_alpha.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_runtime_by_dimension_colored_by_alg(sweep_results_df: pd.DataFrame,
                                             plot_dir: str,
                                             title_str: str = None):
    # Manually make figure bigger to handle external legend
    plt.figure(figsize=(9, 4))

    sns.lineplot(data=sweep_results_df,
                 x='n_features',
                 y='Runtime',
                 hue='inference_alg_str',
                 palette=algorithm_color_map,
                 err_style='bars')
    plt.yscale('log')
    plt.xlabel(r'Data Dimension')

    if title_str is not None:
        plt.title(title_str)

    # Move legend outside of plot
    # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
    plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'runtime_by_dimension.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_scores_by_snr_colored_by_alg(sweep_results_df: pd.DataFrame,
                                      plot_dir: str,
                                      title_str: str = None):
    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:

        # Manually make figure bigger to handle external legend
        plt.figure(figsize=(9, 4))

        sns.lineplot(data=sweep_results_df,
                     x='snr',
                     y=score_column,
                     hue='inference_alg_str',
                     palette=algorithm_color_map)
        plt.xscale('log')
        plt.xlabel(r'SNR')

        # Move legend outside of plot
        # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
        plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        if title_str is not None:
            plt.title(title_str)

        # plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_snr.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_scores_by_alpha_colored_by_alg(sweep_results_df: pd.DataFrame,
                                        plot_dir: str,
                                        title_str: str = None):
    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:

        # Manually make figure bigger to handle external legend
        plt.figure(figsize=(9, 4))

        sns.lineplot(data=sweep_results_df,
                     x='alpha',
                     y=score_column,
                     hue='inference_alg_str',
                     palette=algorithm_color_map)
        plt.xlabel(r'$\alpha$')
        # plt.legend()
        # Move legend outside of plot
        # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
        plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        if title_str is not None:
            plt.title(title_str)

        # plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_alpha.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_scores_by_dimension_colored_by_alg(sweep_results_df: pd.DataFrame,
                                            plot_dir: str,
                                            title_str: str = None):
    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:

        # Manually make figure bigger to handle external legend
        plt.figure(figsize=(9, 4))

        sns.lineplot(data=sweep_results_df,
                     x='n_features',
                     y=score_column,
                     hue='inference_alg_str',
                     palette=algorithm_color_map,
                     err_style="bars", )
        plt.xlabel(r'Data Dimension')
        # plt.legend()

        # Move legend outside of plot
        # See https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
        plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        if title_str is not None:
            plt.title(title_str)

        # plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_dimension.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
