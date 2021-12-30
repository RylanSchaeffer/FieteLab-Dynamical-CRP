import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import os
import torch
from timeit import default_timer as timer

import helpers.numpy
import helpers.torch
import utils.inference
import utils.metrics
import utils.plot
from utils.single_run import *
import data.real
from data.real import *
torch.set_default_tensor_type('torch.FloatTensor')

def main():
    num_data = None
    feature_extractor_method = 'vae'
    center_crop = True
    avg_pool = False
    # plot_dir = 'omniglot_plots_numdata={}_dense'.format(num_data)
    
    plot_dir = '1-omniglot/plots_numdata={}_dense'.format(num_data) # fix as needed
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)
    torch.manual_seed(0)

    omniglot_data = real.load_omniglot_dataset(
        data_dir='data',
        num_data=num_data,
        center_crop=center_crop,
        avg_pool=avg_pool,
        feature_extractor_method=feature_extractor_method)

    # plot number of topics versus number of posts
    plot.plot_num_clusters_by_num_obs(
        true_cluster_labels=omniglot_data['assigned_table_seq'],
        plot_dir=plot_dir)

    num_obs = omniglot_data['assigned_table_seq'].shape[0]
    # num_permutations = 3
    num_permutations = 1
    inference_algs_results_by_dataset_idx = {}
    dataset_by_dataset_idx = {}

    # Generate datasets and record performance for each
    for dataset_idx in range(num_permutations):
        # print(f'Dataset Index: {dataset_idx}')
        # dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        # os.makedirs(dataset_dir, exist_ok=True)

        # generate permutation and reorder data
        # index_permutation = np.random.permutation(np.arange(num_obs, dtype=np.int))
        # omniglot_dataset_results['image_features'] = omniglot_dataset_results['image_features'][index_permutation]
        # omniglot_dataset_results['assigned_table_seq'] = omniglot_dataset_results['assigned_table_seq'][index_permutation]
        dataset_by_dataset_idx[dataset_idx] = dict(
            assigned_table_seq=np.copy(omniglot_data['assigned_table_seq']),
            observations=np.copy(omniglot_data['image_features']))

        dataset_inference_algs_results = single_run(
            dataset_dir=plot_dir,
            sampled_data=omniglot_data,
            setting='omniglot')
        inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results

    plot.plot_inference_algs_comparison(
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
        dataset_by_dataset_idx=dataset_by_dataset_idx,
        plot_dir=plot_dir)

    print('Successfully completed Omniglot simulation')


if __name__ == '__main__':
    main()
