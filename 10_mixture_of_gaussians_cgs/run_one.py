"""
Perform inference in a synthetic mixture of Gaussians for the specified inference
algorithm and model parameters.

Example usage:

10_mixture_of_gaussians_cgs/run_one.py
"""

import joblib
import logging
import numpy as np
import os
import torch
import wandb


import rncrp.data.synthetic
import rncrp.helpers.dynamics
import rncrp.helpers.run
from rncrp.inference import CollapsedGibbsSamplerNew
import rncrp.metrics
import rncrp.plot.plot_general

config_defaults = {
    'inference_alg_str': 'CollapsedGibbsSampler',
    'dynamics_str': 'step',
    'n_samples': 1000,
    'n_features': 2,
    'alpha': 5,
    'beta': 0.,
    'centroids_prior_cov_prefactor': 250.,
    'likelihood_cov_prefactor': 5.,
    'repeat_idx': 0,
}


wandb.init(project='dcrp-mixture-of-gaussians-cgs',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# determine paths
exp_dir = '10_mixture_of_gaussians_cgs'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inf_alg_results_path = os.path.join(
    results_dir_path, f'id={wandb.run.id}.joblib')

wandb.log({'inf_alg_results_path': inf_alg_results_path},
          step=0)

# set seeds
rncrp.helpers.run.set_seed(seed=config['repeat_idx'])

gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': {'a': 1., 'b': 0.}
    },
    'component_prior_params': {
        'centroids_prior_cov_prefactor': config['centroids_prior_cov_prefactor']
    },
    'likelihood_params': {
        'distribution': 'multivariate_normal',
        'likelihood_cov_prefactor': config['likelihood_cov_prefactor']
    }
}


cgs = CollapsedGibbsSamplerNew(
    gen_model_params=gen_model_params,)


gweke_test_results = cgs.geweke_test(
    num_obs=config['n_samples'],
    obs_dim=config['n_features'])


forward_statistics = gweke_test_results['forward_statistics']
gibbs_statistics = gweke_test_results['gibbs_statistics']

import matplotlib.pyplot as plt


plt.plot(np.sort(forward_statistics), np.sort(gibbs_statistics))
min_value = min(np.min(forward_statistics), np.min(gibbs_statistics))
max_value = max(np.max(forward_statistics), np.max(gibbs_statistics))
plt.title('Num Inferred Clusters')
plt.xlabel('Forward')
plt.ylabel('Gibbs Sampling (Median)')
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)
plt.tight_layout()
plt.savefig(os.path.join(exp_dir,
                         f'gweke_test_num_inferred_clusters.png'),
            bbox_inches='tight',
            dpi=300)
# plt.show()
plt.close()

# scores, map_cluster_assignments = rncrp.metrics.compute_predicted_clusters_scores(
#     cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
#     true_cluster_assignments=mixture_model_results['cluster_assignments'])
# inference_alg_results.update(scores)
# inference_alg_results['map_cluster_assignments'] = map_cluster_assignments
# wandb.log(scores, step=0)
#
# sum_sqrd_distances = rncrp.metrics.compute_sum_of_squared_distances_to_nearest_center(
#     X=mixture_model_results['observations'],
#     centroids=inference_alg_results['inference_alg'].centroids_after_last_obs())
# inference_alg_results['training_reconstruction_error'] = sum_sqrd_distances
# wandb.log({'training_reconstruction_error': sum_sqrd_distances}, step=0)

# Need to convert WandB config to proper dict.
data_to_store = dict(
    config=dict(config),
    gweke_test_results=gweke_test_results)

joblib.dump(data_to_store,
            filename=inf_alg_results_path)

print(f'Finished 10_mixture_of_gaussians_cgs/run_one.py for run={wandb.run.id}.')
