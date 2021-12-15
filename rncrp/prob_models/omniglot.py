import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import os
import torch
from timeit import default_timer as timer

import rncrp.helpers.numpy
import rncrp.helpers.torch
import rncrp.inference
import rncrp.metrics
import rncrp.plot
import rncrp.data.real
from rncrp.data.real import *
torch.set_default_tensor_type('torch.FloatTensor')

def single_run(dataset_dir, omniglot_dataset_results):

    concentration_params = np.linspace(0.1*np.log(omniglot_dataset_results['assigned_table_seq'].shape[0]),
                                       3*np.log(omniglot_dataset_results['assigned_table_seq'].shape[0]),
                                       11)
    inference_alg_strs = ['RN-CRP',
                        'DP-Means (online)',
                        'DP-Means (offline)',
                        'DP-GMM (15 Init, 30 Iter)'] # todo: change DP-GMM parameters

    inference_algs_results = {}
    for inference_alg_str in inference_alg_strs:
        inference_alg_results = run_and_plot_inference_alg(
            omniglot_dataset_results=omniglot_dataset_results,
            inference_alg_str=inference_alg_str,
            concentration_params=concentration_params,
            plot_dir=dataset_dir)
        inference_algs_results[inference_alg_str] = inference_alg_results
    return inference_algs_results, sampled_gaussian_data


def run_and_plot_inference_alg(omniglot_dataset_results,
                               inference_alg_str,
                               concentration_params,
                               plot_dir):

    inference_alg_plot_dir = os.path.join(plot_dir, inference_alg_str)
    os.makedirs(inference_alg_plot_dir, exist_ok=True)
    num_clusters_by_concentration_param = {}
    scores_by_concentration_param = {}
    runtimes_by_concentration_param = {}

    for concentration_param in concentration_params:

        inference_alg_results_concentration_param_path = os.path.join(
            inference_alg_plot_dir,
            f'results_{np.round(concentration_param, 2)}.joblib')

        # if results do not exist, generate
        if not os.path.isfile(inference_alg_results_concentration_param_path):
            print(f'Generating {inference_alg_results_concentration_param_path}')

            # run inference algorithm
            # time using timer because https://stackoverflow.com/a/25823885/4570472
            start_time = timer()
            inference_alg_concentration_param_results = utils.inference.run_inference_alg(
                inference_alg_str=inference_alg_str,
                observations=omniglot_dataset_results['image_features'],
                concentration_param=concentration_param,
                likelihood_model='multivariate_normal',
                learning_rate=1e0)

            # record elapsed time
            stop_time = timer()
            runtime = stop_time - start_time

            # record scores
            scores, pred_cluster_labels = utils.metrics.score_predicted_clusters(
                true_cluster_labels=omniglot_dataset_results['assigned_table_seq'],
                table_assignment_posteriors=inference_alg_concentration_param_results['table_assignment_posteriors'])

            # count number of clusters
            num_clusters = len(np.unique(pred_cluster_labels))

            # write to disk and delete
            data_to_store = dict(
                inference_alg_concentration_param_results=inference_alg_concentration_param_results,
                num_clusters=num_clusters,
                scores=scores,
                runtime=runtime,
            )

            joblib.dump(data_to_store,
                        filename=inference_alg_results_concentration_param_path)
            del inference_alg_concentration_param_results
            del data_to_store
        else:
            print(f'Loading {inference_alg_results_concentration_param_path} from disk...')

        # read results from disk
        stored_data = joblib.load(
            inference_alg_results_concentration_param_path)

        num_clusters_by_concentration_param[concentration_param] = stored_data['num_clusters']
        scores_by_concentration_param[concentration_param] = stored_data['scores']
        runtimes_by_concentration_param[concentration_param] = stored_data['runtime']

        print('Finished {} concentration_param={:.2f}'.format(inference_alg_str, concentration_param))

    inference_alg_concentration_param_results = dict(
        num_clusters_by_param=num_clusters_by_concentration_param,
        scores_by_param=pd.DataFrame(scores_by_concentration_param).T,
        runtimes_by_param=runtimes_by_concentration_param,
    )

    return inference_alg_concentration_param_results

def main():
    num_data = None
    feature_extractor_method = 'vae'
    center_crop = True
    avg_pool = False
    plot_dir = 'omniglot_plots_numdata={}_dense'.format(num_data) # fix as needed
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)
    torch.manual_seed(0)

    omniglot_dataset_results = real.load_omniglot_dataset(
        data_dir='data',
        num_data=num_data,
        center_crop=center_crop,
        avg_pool=avg_pool,
        feature_extractor_method=feature_extractor_method)

    # plot number of topics versus number of posts
    plot.plot_num_clusters_by_num_obs(
        true_cluster_labels=omniglot_dataset_results['assigned_table_seq'],
        plot_dir=plot_dir)

    num_obs = omniglot_dataset_results['assigned_table_seq'].shape[0]
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
            assigned_table_seq=np.copy(omniglot_dataset_results['assigned_table_seq']),
            observations=np.copy(omniglot_dataset_results['image_features']))

        dataset_inference_algs_results = single_run(
            dataset_dir=plot_dir,
            omniglot_dataset_results=omniglot_dataset_results)
        inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results

    plot.plot_inference_algs_comparison(
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
        dataset_by_dataset_idx=dataset_by_dataset_idx,
        plot_dir=plot_dir)

    print('Successfully completed Omniglot simulation')


if __name__ == '__main__':
    main()
