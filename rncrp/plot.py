# Common plotting functions
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List


def calculate_num_clusters_by_dataset_by_inference_alg(inference_algs_results_by_dataset_idx):
    # construct dictionary mapping from inference alg to dataframe
    # with dataset idx as rows and concentration parameters as columns
    # {inference alg: DataFrame(number of clusters)}
    num_clusters_by_dataset_by_inference_alg = {}
    for dataset_inf_algs_results in inference_algs_results_by_dataset_idx:
        for inf_alg_str, inf_alg_results in dataset_inf_algs_results.items():
            inf_alg_family = '{}_dyn={}'.format(
                inf_alg_results['inference_alg_str'],
                inf_alg_results['inference_dynamics_str'])
            if inf_alg_family not in num_clusters_by_dataset_by_inference_alg:
                num_clusters_by_dataset_by_inference_alg[inf_alg_family] = dict()
            alpha = inf_alg_results['inference_alg_params']['alpha']
            if alpha not in num_clusters_by_dataset_by_inference_alg[inf_alg_family]:
                num_clusters_by_dataset_by_inference_alg[inf_alg_family][alpha] = []
            num_clusters_by_dataset_by_inference_alg[inf_alg_family][alpha].append(
                inf_alg_results['num_clusters'])
    for inf_alg in num_clusters_by_dataset_by_inference_alg:
        # we need to orient then transpose in case number of clusters per different concentration
        # parameter have different lengths
        # https://stackoverflow.com/questions/54657896/fill-with-default-0s-when-creating-a-dataframe-in-pandas
        num_clusters_by_dataset_by_inference_alg[inf_alg] = pd.DataFrame.from_dict(
            num_clusters_by_dataset_by_inference_alg[inf_alg],
            orient='index').T

    return num_clusters_by_dataset_by_inference_alg

def calculate_runtimes_by_dataset_by_inference_alg(inference_algs_results_by_dataset_idx):
    # construct dictionary mapping from inference alg to dataframe
    # with dataset idx as rows and concentration parameters as columns
    # {inference alg: DataFrame(runtimes)}
    runtimes_by_dataset_by_inference_alg = {}
    for dataset_inf_algs_results in inference_algs_results_by_dataset_idx:
        for inf_alg_str, inf_alg_results in dataset_inf_algs_results.items():
            inf_alg_family = '{}_dyn={}'.format(
                inf_alg_results['inference_alg_str'],
                inf_alg_results['inference_dynamics_str'])
            if inf_alg_family not in runtimes_by_dataset_by_inference_alg:
                runtimes_by_dataset_by_inference_alg[inf_alg_family] = dict()
            alpha = inf_alg_results['inference_alg_params']['alpha']
            if alpha not in runtimes_by_dataset_by_inference_alg[inf_alg_family]:
                runtimes_by_dataset_by_inference_alg[inf_alg_family][alpha] = []
            runtimes_by_dataset_by_inference_alg[inf_alg_family][alpha].append(
                inf_alg_results['runtime'])
    for inf_alg in runtimes_by_dataset_by_inference_alg:
        # we need to orient then transpose in case runtimes per different concentration
        # parameter have different lengths
        # https://stackoverflow.com/questions/54657896/fill-with-default-0s-when-creating-a-dataframe-in-pandas
        runtimes_by_dataset_by_inference_alg[inf_alg] = pd.DataFrame.from_dict(
            runtimes_by_dataset_by_inference_alg[inf_alg],
            orient='index').T

    return runtimes_by_dataset_by_inference_alg


def calculate_scores_by_dataset_by_inference_alg_by_scoring_metric(inference_algs_results_by_dataset_idx,
                                                                   scoring_metric_strs: List[str]):
    # construct dictionary mapping from scoring metric to inference alg
    # to dataframe with dataset idx as rows and concentration parameters as columns
    # {scoring metric: {inference alg: DataFrame(scores)}}

    scores_by_dataset_by_inference_alg_by_scoring_metric = {}
    for scoring_metric_str in scoring_metric_strs:
        scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric_str] = {}
        for dataset_inf_algs_results in inference_algs_results_by_dataset_idx:
            for inf_alg_str, inf_alg_results in dataset_inf_algs_results.items():
                inf_alg_family = '{}_dyn={}'.format(
                    inf_alg_results['inference_alg_str'],
                    inf_alg_results['inference_dynamics_str'])
                if inf_alg_family not in scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric_str]:
                    scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric_str][inf_alg_family] = dict()
                alpha = inf_alg_results['inference_alg_params']['alpha']
                if alpha not in scores_by_dataset_by_inference_alg_by_scoring_metric[
                    scoring_metric_str][inf_alg_family]:
                    scores_by_dataset_by_inference_alg_by_scoring_metric[
                        scoring_metric_str][inf_alg_family][alpha] = []
                # assert isinstance(inf_alg_results['scores'][scoring_metric_str], np.float64)
                # assert isinstance(scores_by_dataset_by_inference_alg_by_scoring_metric[inf_alg_family][alpha], list)
                scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric_str][inf_alg_family][alpha].append(
                    inf_alg_results['scores'][scoring_metric_str])
                # assert isinstance(inf_alg_results['scores'][scoring_metric_str], np.float64)
                # assert isinstance(scores_by_dataset_by_inference_alg_by_scoring_metric[inf_alg_family][alpha], list)

        for inf_alg in scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric_str]:
            # we need to orient then transpose in case scores per different concentration
            # parameter have different lengths
            # https://stackoverflow.com/questions/54657896/fill-with-default-0s-when-creating-a-dataframe-in-pandas
            scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric_str][inf_alg] = \
                pd.DataFrame.from_dict(
                    scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric_str][inf_alg],
                    orient='index').T

    return scores_by_dataset_by_inference_alg_by_scoring_metric


def plot_inference_algs_comparison(inference_algs_results_by_dataset_idx: List[Dict[str, dict]],
                                   dataset_by_dataset_idx: List[dict],
                                   plot_dir: str):
    # exclude datasets with no inference results
    dataset_by_dataset_idx = [dataset for dataset_idx, dataset in enumerate(dataset_by_dataset_idx)
                              if len(inference_algs_results_by_dataset_idx[dataset_idx]) > 0]
    inference_algs_results_by_dataset_idx = [inf_algs_results for dataset_idx, inf_algs_results
                                             in enumerate(inference_algs_results_by_dataset_idx)
                                             if len(inf_algs_results) > 0]

    if len(inference_algs_results_by_dataset_idx) == 0:
        return

    num_datasets = len(inference_algs_results_by_dataset_idx)
    num_clusters_by_dataset_idx = [len(np.unique(dataset['assigned_table_seq']))
                                   for dataset in dataset_by_dataset_idx]

    # inference_alg_families = np.unique([
    #     inference_algs_results_by_dataset_idx[dataset_idx][inf_alg_str]['inference_alg_str']
    #     for dataset_idx in range(num_datasets)
    #     for inf_alg_str in inference_algs_results_by_dataset_idx[dataset_idx]])

    scoring_metric_strs = list(inference_algs_results_by_dataset_idx[0][
                                   list(inference_algs_results_by_dataset_idx[0].keys())[0]]['scores'].keys())

    # we have four dimensions of interest:
    #   inference_alg
    #   dataset idx
    #   scoring metrics
    #   concentration parameters

    num_clusters_by_dataset_by_inference_alg = calculate_num_clusters_by_dataset_by_inference_alg(
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx)

    plot_inference_algs_num_clusters_by_param(
        num_clusters_by_dataset_by_inference_alg=num_clusters_by_dataset_by_inference_alg,
        plot_dir=plot_dir,
        num_clusters_by_dataset_idx=num_clusters_by_dataset_idx)

    runtimes_by_dataset_by_inference_alg = calculate_runtimes_by_dataset_by_inference_alg(
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx)

    plot_inference_algs_runtimes_by_param(
        runtimes_by_dataset_by_inference_alg=runtimes_by_dataset_by_inference_alg,
        plot_dir=plot_dir)

    scores_by_dataset_by_inference_alg_by_scoring_metric = \
        calculate_scores_by_dataset_by_inference_alg_by_scoring_metric(
            inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
            scoring_metric_strs=scoring_metric_strs)

    plot_inference_algs_scores_by_param(
        scores_by_dataset_by_inference_alg_by_scoring_metric=scores_by_dataset_by_inference_alg_by_scoring_metric,
        plot_dir=plot_dir)


def plot_inference_algs_num_clusters_by_param(num_clusters_by_dataset_by_inference_alg: dict,
                                              plot_dir: str,
                                              num_clusters_by_dataset_idx: List[int]):
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(6, 6))

    dfs = []
    for inference_alg_str, inference_alg_num_clusters_df in num_clusters_by_dataset_by_inference_alg.items():
        # make columns of concentration parameters into column
        inference_alg_num_clusters_df['alg'] = inference_alg_str
        dfs.append(inference_alg_num_clusters_df)

    dfs = pd.concat(dfs)
    melted_df = dfs.melt(
        id_vars=['alg'],  # columns to keep
        var_name='alpha',  # new column name for previous columns headers
        value_name='num_clusters',  # new column name for values
    )

    g = sns.pointplot(data=melted_df,
                      x='alpha',
                      y='num_clusters',
                      hue='alg',
                      ax=ax)

    # rotate xlabels
    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    ax.set_xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    ax.set_ylabel('Number of Clusters')
    plt.axhline(np.mean(num_clusters_by_dataset_idx),
                label='Correct Number Clusters', linestyle='--', color='k')
    ax.set_ylim(bottom=1.)

    ax.set_yscale('log')
    plt.savefig(os.path.join(plot_dir, f'num_clusters_by_param.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_inference_algs_scores_by_param(scores_by_dataset_by_inference_alg_by_scoring_metric: dict,
                                        plot_dir: str):
    # for each scoring function, plot score (y) vs parameter (x)
    for scoring_metric, scores_by_dataset_by_inference_alg in \
            scores_by_dataset_by_inference_alg_by_scoring_metric.items():

        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               figsize=(6, 6))

        dfs = []
        for inference_alg_str, inference_algs_scores_df in scores_by_dataset_by_inference_alg.items():
            # make columns of concentration parameters into column
            inference_algs_scores_df['alg'] = inference_alg_str
            dfs.append(inference_algs_scores_df)

        dfs = pd.concat(dfs)
        # make columns of concentration parameters into column
        melted_df = dfs.melt(
            id_vars=['alg'],  # columns to keep
            var_name='alpha',  # new column name for previous columns headers
            value_name='score',  # new column name for values
        )

        sns.pointplot(data=melted_df,
                      x='alpha',
                      y='score',
                      hue='alg',
                      label=inference_alg_str,
                      ax=ax)

        ax.set_xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
        ax.set_ylabel(scoring_metric)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'comparison_score={scoring_metric}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_inference_algs_runtimes_by_param(runtimes_by_dataset_by_inference_alg: dict,
                                          plot_dir: str):
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(6, 6))

    dfs = []
    for inference_alg_str, inference_alg_runtimes_df in runtimes_by_dataset_by_inference_alg.items():
        # make columns of concentration parameters into column
        inference_alg_runtimes_df['alg'] = inference_alg_str
        dfs.append(inference_alg_runtimes_df)

    dfs = pd.concat(dfs)
    # make columns of concentration parameters into column
    melted_df = dfs.melt(
        id_vars=['alg'],  # columns to keep
        var_name='alpha',  # new column name for previous columns headers
        value_name='runtime',  # new column name for values
    )

    g = sns.pointplot(data=melted_df,
                      x='alpha',
                      y='runtime',
                      hue='alg',
                      label=inference_alg_str,
                      ax=ax)
    # rotate xlabels
    plt.setp(g.get_xticklabels(), rotation=45)

    ax.set_xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    plt.savefig(os.path.join(plot_dir, 'runtimes_by_param.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_num_clusters_by_num_obs(true_cluster_labels,
                                 plot_dir: str):
    # compute the empirical number of topics per number of posts
    unique_labels = set()
    empiric_num_unique_clusters_by_end_index = []
    for cluster_label in true_cluster_labels:
        unique_labels.add(cluster_label)
        empiric_num_unique_clusters_by_end_index.append(len(unique_labels))
    empiric_num_unique_clusters_by_end_index = np.array(empiric_num_unique_clusters_by_end_index)

    obs_indices = 1 + np.arange(len(empiric_num_unique_clusters_by_end_index))

    # fit alpha to the empirical number of topics per number of posts
    def expected_num_tables(customer_idx, alpha):
        return np.multiply(alpha, np.log(1 + customer_idx / alpha))

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(f=expected_num_tables,
                           xdata=obs_indices,
                           ydata=empiric_num_unique_clusters_by_end_index)
    fitted_alpha = popt[0]

    fitted_num_unique_clusters_by_end_index = expected_num_tables(
        customer_idx=obs_indices,
        alpha=fitted_alpha)

    plt.plot(obs_indices,
             empiric_num_unique_clusters_by_end_index,
             label='Empiric')
    plt.plot(obs_indices,
             fitted_num_unique_clusters_by_end_index,
             label=f'Fit (alpha = {np.round(fitted_alpha, 2)})')
    plt.xlabel('Observation Index')
    plt.ylabel('New Cluster (Ground Truth)')
    plt.legend()

    # make axes equal
    # plt.axis('square')

    plt.savefig(os.path.join(plot_dir, 'fitted_alpha.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
