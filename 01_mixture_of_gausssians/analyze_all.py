"""
Analyze results generated by run_all.py (or equivalently, run_all.sh on
a SLURM cluster).
"""

import itertools
import joblib
import logging
import numpy as np
import os
from typing import Dict, List, Tuple


import utils.data
import utils.plot


def analyze_all():
    # create directory
    exp_dir_path = '01_mixture_of_gaussians'
    results_dir_path = os.path.join(exp_dir_path, 'results')

    dataset_by_dataset_idx_by_sampling, inference_algs_results_by_dataset_idx_by_sampling = \
        construct_data_and_inference_results_by_sampling(results_dir_path=results_dir_path)

    # inference_results_stats_by_sampling = construct_inference_results_stats_by_sampling(
    #     data_and_inference_results_by_sampling=data_and_inference_results_by_sampling)

    for sampling_dir_name in os.listdir(results_dir_path):
        sampling_dir_path = os.path.join(results_dir_path, sampling_dir_name)
        if len(inference_algs_results_by_dataset_idx_by_sampling[sampling_dir_name]) == 0:
            continue
        sampling_dir_results_path = os.path.join(sampling_dir_path, 'results')
        os.makedirs(sampling_dir_results_path, exist_ok=True)
        utils.plot.plot_inference_algs_comparison(
            inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx_by_sampling[sampling_dir_name],
            dataset_by_dataset_idx=dataset_by_dataset_idx_by_sampling[sampling_dir_name],
            plot_dir=sampling_dir_results_path)


def construct_data_and_inference_results_by_sampling(
        results_dir_path: str) -> Tuple[Dict[str, List[dict]],
                                        Dict[str, List[Dict[str, dict]]]]:
    """
    Returns a 2-tuple of datasets and inference algorithm results.

    dataset_by_dataset_idx_by_sampling has the structure:
        {how data was sampled: [dataset_results]}

    inference_algs_results_by_dataset_idx_by_sampling has the structure:
        {how data was sampled: [{inference_alg_str: inference_alg_results}]}
    """

    inference_algs_results_by_dataset_idx_by_sampling = {}
    dataset_by_dataset_idx_by_sampling = {}

    for sampling_dir_name in os.listdir(results_dir_path):
        sampling_dir_path = os.path.join(results_dir_path, sampling_dir_name)
        inference_algs_results_by_dataset_idx_by_sampling[sampling_dir_name] = []
        dataset_by_dataset_idx_by_sampling[sampling_dir_name] = []
        for dataset_dir_name in sorted(os.listdir(sampling_dir_path)):
            dataset_dir_path = os.path.join(sampling_dir_path, dataset_dir_name)
            if not os.path.isdir(dataset_dir_path):
                continue
            try:
                data_results = joblib.load(
                    os.path.join(dataset_dir_path, 'data.joblib'))
            except FileNotFoundError:
                continue
            dataset_by_dataset_idx_by_sampling[sampling_dir_name].append(data_results)
            inference_algs_results = dict()
            for file_name in os.listdir(dataset_dir_path):
                inference_alg_results_path = os.path.join(dataset_dir_path, file_name)
                if os.path.isdir(inference_alg_results_path):
                    try:
                        inference_alg_results = joblib.load(
                            os.path.join(inference_alg_results_path, 'inference_results.joblib'))
                        inference_algs_results[file_name] = inference_alg_results
                    except FileNotFoundError:
                        continue
            inference_algs_results_by_dataset_idx_by_sampling[sampling_dir_name].append(
                inference_algs_results)

    return dataset_by_dataset_idx_by_sampling, inference_algs_results_by_dataset_idx_by_sampling


# def construct_inference_results_stats_by_sampling(data_and_inference_results_by_sampling):
#     inference_results_stats_by_sampling = dict()
#     for sampling_dir_name, sampling_data_and_inference_results in data_and_inference_results_by_sampling.items():
#         stats = []
#         for dataset_idx, dataset_results in enumerate(sampling_data_and_inference_results):
#             sampling_scheme = dataset_results['data_results']['cluster_assignment_sampling']
#             true_num_clusters = len(np.unique(dataset_results['data_results']['assigned_table_seq']))
#             for inference_alg_results in dataset_results['inference_algs'].values():
#                 try:
#                     inference_alg_str = inference_alg_results['inference_alg_str']
#                     inferred_num_clusters = inference_alg_results['num_clusters']
#                     runtime = inference_alg_results['runtime']
#                     row = dict(
#                         sampling_scheme=sampling_scheme,
#                         dataset_idx=dataset_idx,
#                         true_num_clusters=true_num_clusters,
#                         inference_alg_str=inference_alg_str,
#                         inferred_num_clusters=inferred_num_clusters,
#                         runtime=runtime,
#                     )
#
#                     for key, value in inference_alg_results['scores'].items():
#                         row[key] = value
#                     for key, value in dataset_results['data_results']['cluster_assignment_sampling_parameters'].items():
#                         row[key] = str(value)
#                     for key, value in inference_alg_results['inference_alg_params'].items():
#                         row[key] = str(value)
#
#                     stats.append(row)
#
#                 except KeyError:
#                     continue
#         stats = pd.DataFrame(stats)
#         inference_results_stats_by_sampling[sampling_dir_name] = stats
#
#     return inference_results_stats_by_sampling


# def construct_data_and_inference_results_by_sampling(
#         results_dir_path: str) -> Dict[str, List[Dict[str, Dict]]]:
#     """
#     Construct the following structure from data on disk
#         {
#             how data was sampled: [
#                 dataset_idx: {
#                     data: {data}
#                     inference_algs: {
#                         inference_alg_str: {results}
#                     }
#                 }
#             ]
#         }
#     """
#
#     data_and_inference_results_by_sampling = dict()
#     for sampling_dir_name in os.listdir(results_dir_path):
#         sampling_dir_path = os.path.join(results_dir_path, sampling_dir_name)
#         data_and_inference_results_by_sampling[sampling_dir_name] = []
#         for dataset_dir_name in sorted(os.listdir(sampling_dir_path)):
#             dataset_dir_path = os.path.join(sampling_dir_path, dataset_dir_name)
#             data_results = joblib.load(
#                 os.path.join(dataset_dir_path, 'data.joblib'))
#             data_and_inference_results = dict(
#                 data_results=data_results,
#                 inference_algs=dict())
#
#             for file_name in os.listdir(dataset_dir_path):
#                 inference_alg_results_path = os.path.join(dataset_dir_path, file_name)
#                 if os.path.isdir(inference_alg_results_path):
#                     inference_alg_results = joblib.load(
#                         os.path.join(inference_alg_results_path, 'inference_results.joblib'))
#                     data_and_inference_results['inference_algs'][file_name] = inference_alg_results
#
#             data_and_inference_results_by_sampling[sampling_dir_name].append(
#                 data_and_inference_results)
#
#     return data_and_inference_results_by_sampling


if __name__ == '__main__':
    analyze_all()