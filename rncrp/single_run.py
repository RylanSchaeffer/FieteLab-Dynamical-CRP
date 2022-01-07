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
import data.real
from data.real import *
from data.synthetic import *

torch.set_default_tensor_type('torch.FloatTensor')


def single_run(dataset_dir, sampled_data, setting):
    if setting == 'omniglot':
        concentration_params = np.linspace(0.1 * np.log(sampled_data['assigned_table_seq'].shape[0]),
                                           3 * np.log(sampled_data['assigned_table_seq'].shape[0]),
                                           11)

    elif setting == 'gaussian':
        concentration_params = 0.01 + np.arange(0., 6.01, 0.25)  # todo: select other values as needed

    inference_alg_strs = ['RN-CRP',
                          'DP-Means (online)',
                          'DP-Means (offline)',
                          'DP-GMM (15 Init, 30 Iter)']  # todo: change DP-GMM parameters

    inference_algs_results = {}
    for inference_alg_str in inference_alg_strs:
        inference_alg_results = run_and_plot_inference_alg(
            sampled_data=sampled_data,
            inference_alg_str=inference_alg_str,
            concentration_params=concentration_params,
            plot_dir=dataset_dir)
        inference_algs_results[inference_alg_str] = inference_alg_results
    return inference_algs_results, sampled_gaussian_data


def run_and_plot_inference_alg(sampled_data,
                               inference_alg_str,
                               concentration_params,
                               plot_dir):
    inference_alg_plot_dir = os.path.join(plot_dir, inference_alg_str)
    os.makedirs(inference_alg_plot_dir, exist_ok=True)
    num_clusters_by_concentration_param = {}
    scores_by_concentration_param = {}
    runtimes_by_concentration_param = {}

    if setting == 'omniglot':
        features = 'image_features'
        likelihood = 'multivariate_normal'
        # likelihood = 'dirichlet_multinomial'

    elif setting == 'gaussian':
        features = 'gaussian_samples_seq'
        likelihood = 'multivariate_normal'

    for concentration_param in concentration_params:

        inference_alg_results_concentration_param_path = os.path.join(
            inference_alg_plot_dir,
            f'results_{np.round(concentration_param, 2)}.joblib')

        # if results do not exist, generate
        if not os.path.isfile(inference_alg_results_concentration_param_path):
            print(f'Generating {inference_alg_results_concentration_param_path}')

            # run inference algorithm
            start_time = timer()
            inference_alg_concentration_param_results = utils.inference.run_inference_alg(
                inference_alg_str=inference_alg_str,
                observations=sampled_data[features],
                gen_model_params=concentration_param,
                likelihood_model=likelihood,
                learning_rate=1e0)

            # record elapsed time
            stop_time = timer()
            runtime = stop_time - start_time

            # record scores
            scores, pred_cluster_labels = utils.metrics.compute_predicted_clusters_scores(
                true_cluster_labels=sampled_data['assigned_table_seq'],
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
