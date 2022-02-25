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

# import plot
import rncrp.data.real_nontabular
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics

config_defaults = {
    'inference_alg_str': 'Dynamical-CRP',
    'dynamics_str': 'exp',
    'dynamics_a': 1.,
    'dynamics_b': 1.,
    'alpha': 1.1,
    'beta': 0.,
    'beta_arg1': 1.,
    'beta_arg2': 5.,
    'repeat_idx': 0,
    'narrow_hallways': True,
    'finite_vision': True,
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

yilun_nav_2d_dataset = rncrp.data.real_nontabular.load_dataset_yilun_nav_2d_2022(
    narrow_hallways=config['narrow_hallways'],
    finite_vision=config['finite_vision'],
)


observations = yilun_nav_2d_dataset['vis_matrix']

# Take the repeat-index's environment
env_idx = config['repeat_idx']
observations = observations[env_idx, :, :]

# Construct observation times
# Points has shape (num obs, trajectory length, 2 for xy coord)
num_obs = yilun_nav_2d_dataset['points'].shape[1]
observation_times = np.arange(num_obs)

# Compute true cluster assignments of this environment
true_cluster_assignments = yilun_nav_2d_dataset['room_ids'][env_idx, :]

# Compute number of true clusters
wandb.log({'n_clusters': len(np.unique(true_cluster_assignments))}, step=0)

gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': {'a': config['dynamics_a'], 'b': 1.0},
    },
    'component_prior_params': {
        'beta_arg1': config['beta_arg1'],
        'beta_arg2': config['beta_arg2'],
    },
    'likelihood_params': {
        'distribution': 'product_bernoullis',
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

print('Finished run.')
