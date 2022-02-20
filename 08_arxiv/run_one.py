"""
Perform inference on Arxiv 2022 titles and abstracts dataset,
for the specified inference algorithm and model parameters.

Example usage:

08_arxiv_2021/run_one.py
"""

import joblib
import os

import numpy as np
import pandas as pd
import wandb

import rncrp.data.real_tabular
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
    'n_samples': 10,
    'repeat_idx': 0,
    'imagenet_split': 'val',
    'beta_arg1': 1.,
    'beta_arg2': 1.,
}

wandb.init(project='dcrp-arxiv-2022',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# determine paths
exp_dir = '08_arxiv'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inf_alg_results_path = os.path.join(results_dir_path,
                                    f'id={wandb.run.id}.joblib')
wandb.log({'inf_alg_results_path': inf_alg_results_path},
          step=0)

# set seeds
rncrp.helpers.run.set_seed(seed=config['repeat_idx'])

preprocessed_arxiv_data_path = os.path.join(
    'data', 'arxiv_2022', 'arxiv-metadata.joblib')

if not os.path.isfile(preprocessed_arxiv_data_path):

    # Load raw Arxiv dataset
    arxiv_2022_dataset = rncrp.data.real_tabular.load_dataset_arxiv_2022()

    # Convert Arxiv topics to integer codes
    true_cluster_assignments, unique_cluster_ids = pd.factorize(
        arxiv_2022_dataset['labels'])

    # Massage Arxiv data to nice format
    observations = arxiv_2022_dataset['observations']['text'].copy()
    observation_times = observations['latest_date'].values

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    observations = vectorizer.fit_transform(observations)

    joblib.dump(
        dict(observations=observations,
             observation_times=observation_times,
             true_cluster_assignments=true_cluster_assignments,
             unique_cluster_ids=unique_cluster_ids,
             vectorizer=vectorizer,
             ),
        filename=preprocessed_arxiv_data_path)

else:
    with joblib.load(preprocessed_arxiv_data_path) as joblib_file_ptr:
        observations = joblib_file_ptr['observations']
        observation_times = joblib_file_ptr['observation_times']
        true_cluster_assignments = joblib_file_ptr['true_cluster_assignments']
        unique_cluster_ids = joblib_file_ptr['unique_cluster_ids']
        vectorizer = joblib_file_ptr['vectorizer']


# Take only the first n_samples samples
observations = observations[:config['n_samples']]
observation_times = observation_times[:config['n_samples']]
true_cluster_assignments = true_cluster_assignments[:config['n_samples']]


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
        'distribution': 'dirichlet_multinomial',
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
