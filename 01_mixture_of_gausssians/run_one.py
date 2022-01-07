"""
Perform inference in a mixture of Gaussians for the specified inference
algorithm, dynamics string and time sampling string.

Example usage:

01_mixture_of_gaussians/run_one.py --run_one_results_dir=01_mixture_of_gaussians/results/categorical_probs=[0.2,0.2,0.2,0.2,0.2]/dataset=1 \
 --inference_alg_str=D-CRP \
 --alpha=30.91 --beta=0.0 --dynamics_str=harmonicoscillator
"""

import argparse
import joblib
import logging
import numpy as np
import os
from timeit import default_timer as timer
import torch
import wandb

# import plot
import rncrp.data.synthetic
import rncrp.helpers.dynamics
import rncrp.helpers.run
# import rncrp.inference
import rncrp.metrics


config_defaults = {
    'inference_alg': 'RNCRP',
    'dynamics_str': 'step',
    'n_samples': 1000,
    'n_features': 10,
    'n_clusters': 25,
    'alpha': 1.,
    'beta': 0.,
    'centroids_prior_cov_prefactor': 5.,
    'likelihood_icov_prefactor': 1.,
    'repeat_idx': 0,
}
wandb.init(project='rncrp-mixture-of-gaussians',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)


exp_dir = '01_mixture_of_gaussians'
results_dir_path = os.path.join(exp_dir, 'results')


# set seeds
rncrp.helpers.run.set_seed(config['repeat_idx'])


mixture_model_results = rncrp.data.synthetic.sample_mixture_model(
    num_obs=config['n_samples'],
    obs_dim=config['n_features'],
    mixing_prior_str='rncrp',
    mixing_distribution_params={'alpha': config['alpha'],
                                'beta': config['beta'],
                                'dynamics_str': config['dynamics_str']},
    component_prior_str='gaussian',
    component_prior_params={'centroids_prior_cov_prefactor': config['centroids_prior_cov_prefactor'],
                            'likelihood_cov_prefactor': config['likelihood_cov_prefactor']})

# time using timer because https://stackoverflow.com/a/25823885/4570472
start_time = timer()
dpmeans.fit(X=mixture_model_results['obs'])
stop_time = timer()
runtime = stop_time - start_time

results = {
    'Runtime': runtime,
    'Num Iter Till Convergence': dpmeans.n_iter_,
    'Num Initial Clusters': dpmeans.num_init_clusters_,
    'Num Inferred Clusters': dpmeans.num_clusters_,
    'Loss': dpmeans.loss_,
}

scores, pred_cluster_assignments = compute_predicted_clusters_scores(
    cluster_assignment_posteriors=dpmeans.labels_,
    true_cluster_assignments=mixture_model_results['cluster_assignments'],
)
results.update(scores)

wandb.log(results, step=0)

data_to_store = dict(
    inference_alg_str=inference_alg_str,
    inference_dynamics_str=inference_dynamics_str,
    inference_alg_params=inference_alg_params,
    inference_results=inference_results,
    num_clusters=num_clusters,
    scores=scores,
    runtime=runtime)

joblib.dump(data_to_store,
            filename=inference_results_path)

# read results from disk
stored_data = joblib.load(inference_results_path)

plot.plot_inference_results(
    sampled_mog_data=sampled_mog_data,
    inference_results=stored_data['inference_results'],
    inference_alg_str=stored_data['inference_alg_str'],
    inference_alg_param=stored_data['inference_alg_params'],
    plot_dir=inference_results_dir)

run_and_plot_inference_alg(
    sampled_mog_data=setup_results['sampled_mog_data'],
    inference_alg_str=setup_results['inference_alg_str'],
    inference_alg_params=setup_results['inference_alg_params'],
    inference_dynamics_str=setup_results['dynamics_str'],
    inference_results_dir=setup_results['inference_results_dir'])


print('Finished run.')


