import numpy as np
import sklearn.mixture
from typing import Dict

from rncrp.inference.base import BaseModel


class VariationalInferenceGMM(BaseModel):
    """
    Variational Inference for Dirichlet Process Mixtures by Blei and Jordan (2006).

    Wrapper around scikit-learn's implementation.
    """
    def __init__(self,
                 model_str: str,
                 gen_model_params: Dict[str, Dict[str, float]],
                 plot_dir: str = None,
                 max_iter: int = 8,  # same as DP-Means
                 num_initializations: int = 1
                 ):

        self.gen_model_params = gen_model_params
        self.rncrp_params = gen_model_params['rncrp']
        assert self.rncrp_params['alpha'] > 0
        assert self.rncrp_params['beta'] > 0
        self.feature_prior_params = gen_model_params['feature_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        self.max_iter = max_iter
        self.num_initializations = num_initializations
        self.model_str = model_str
        self.plot_dir = plot_dir

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray):

            num_obs, obs_dim = observations.shape
            var_dp_gmm = sklearn.mixture.BayesianGaussianMixture(
                n_components=num_obs,
                max_iter=self.max_iter,
                n_init=self.num_initializations,
                init_params='random',
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=self.rncrp_params['alpha'])
            var_dp_gmm.fit(observations)
            cluster_assgt_posteriors = var_dp_gmm.predict_proba(observations)
            cluster_assgt_posteriors_running_sum = np.cumsum(cluster_assgt_posteriors,
                                                             axis=0)
            params = dict(means=var_dp_gmm.means_,
                          covs=var_dp_gmm.covariances_)

            # returns classes assigned and centroids of corresponding classes
            variational_results = dict(
                cluster_assgt_posteriors=cluster_assgt_posteriors,
                cluster_assgt_posteriors_running_sum=cluster_assgt_posteriors_running_sum,
                parameters=params,
            )
            return variational_results


