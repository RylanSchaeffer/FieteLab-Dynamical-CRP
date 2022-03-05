"""
Perform inference on Omniglot VAE embeddings
with the specified inference algorithm and model parameters.

Example usage:

06_omniglot/run_one.py
"""

import joblib
import numpy as np
import os
import wandb

import rncrp.data.real_nontabular
import rncrp.data.synthetic
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics

config_defaults = {
    'inference_alg_str': 'Dynamical-CRP',
    'dynamics_str': 'sinusoid',
    'dynamics_omega': 1.,
    'alpha': 1.1,
    'beta': 0.,
    'n_samples': 500,
    'repeat_idx': 0,
}

wandb.init(project='dcrp-omniglot',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# determine paths
exp_dir = '06_omniglot'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inf_alg_results_path = os.path.join(results_dir_path,
                                    f'id={wandb.run.id}.joblib')
wandb.log({'inf_alg_results_path': inf_alg_results_path},
          step=0)

# set seeds
rncrp.helpers.run.set_seed(seed=config['repeat_idx'])

omniglot_data = rncrp.data.real_nontabular.load_dataset_omniglot_vae(
    data_dir='data')

# Sample cluster IDs to use for slicing
monte_carlo_rncrp_results = rncrp.data.synthetic.sample_dcrp(
    num_mc_samples=1,
    num_customer=config['n_samples'],
    alpha=config['alpha'],
    beta=0.,
    dynamics_str='sinusoid',
    dynamics_params={'omega': config['dynamics_omega']}
)
true_cluster_assignments = monte_carlo_rncrp_results[
    'customer_assignments_by_customer'][0]

# Log number of true clusters
wandb.log({'n_clusters': len(np.unique(true_cluster_assignments))}, step=0)

rncrp.helpers.run.select_indices_given_desired_cluster_assignments_and_labels(
    desired_cluster_assignments=true_cluster_assignments,
    labels=omniglot_data['labels'],
)

observations = omniglot_data['images']


# Construct observation times
num_obs = observations.shape[0]
observation_times = np.arange(num_obs)

gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': {'omega': config['dynamics_omega']},
    },
    'component_prior_params': {
    },
    'likelihood_params': {
        'distribution': 'multivariate_normal',
    }
}

inference_alg_results = rncrp.helpers.run.run_inference_alg(
    inference_alg_str=config['inference_alg_str'],
    observations=observations,
    observations_times=observation_times,
    gen_model_params=gen_model_params,
)

scores, map_cluster_assignments = rncrp.metrics.compute_predicted_clusters_scores(
    cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
    true_cluster_assignments=true_cluster_assignments,
)
inference_alg_results.update(scores)
inference_alg_results['map_cluster_assignments'] = map_cluster_assignments
wandb.log(scores, step=0)

data_to_store = dict(
    config=dict(config),  # Need to convert WandB config to proper dict
    inference_alg_results=inference_alg_results,
    scores=scores)

joblib.dump(data_to_store,
            filename=inf_alg_results_path)

print('Finished 06_omniglot/run_one.py.')
