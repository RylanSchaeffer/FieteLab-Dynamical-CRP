import numpy as np
import sklearn.mixture
from typing import Dict

from rncrp.inference.base import BaseModel


class VariationalInferenceGMM(BaseModel):
    """
    Variational Inference for Dirichlet Process Gaussian Mixture Model, as
    proposed by Blei and Jordan (2006).

    Wrapper around scikit-learn's implementation.
    """

    def __init__(self,
                 gen_model_params: Dict[str, Dict[str, float]],
                 model_str: str = 'VI-GMM',
                 plot_dir: str = None,
                 max_iter: int = 8,  # same as DP-Means
                 num_initializations: int = 1,
                 **kwargs,
                 ):
        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.feature_prior_params = gen_model_params['feature_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        self.max_iter = max_iter
        self.num_initializations = num_initializations
        self.model_str = model_str
        self.plot_dir = plot_dir
        self.fit_results = None

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
            weight_concentration_prior=self.mixing_params['alpha'])
        var_dp_gmm.fit(observations)
        cluster_assignment_posteriors = var_dp_gmm.predict_proba(observations)
        cluster_assignment_posteriors_running_sum = np.cumsum(cluster_assignment_posteriors,
                                                              axis=0)
        params = dict(means=var_dp_gmm.means_,
                      covs=var_dp_gmm.covariances_)

        self.fit_results = dict(
            cluster_assignment_posteriors=cluster_assignment_posteriors,
            cluster_assignment_posteriors_running_sum=cluster_assignment_posteriors_running_sum,
            parameters=params,
        )
        return self.fit_results
