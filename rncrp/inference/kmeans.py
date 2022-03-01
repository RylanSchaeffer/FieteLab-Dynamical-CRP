import numpy as np
from sklearn.cluster import KMeans
from typing import Dict

from rncrp.inference.base import BaseModel


class KMeansWrapper(BaseModel):
    """
    KMeans. Wrapper around scikit-learn's implementation.
    """

    def __init__(self,
                 gen_model_params: Dict[str, Dict[str, float]],
                 model_str: str = 'VI-GMM',
                 plot_dir: str = None,
                 max_iter: int = 100,
                 num_initializations: int = 10,
                 **kwargs,
                 ):
        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.component_prior_params = gen_model_params['component_prior_params']
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
        var_dp_gmm = KMeans(
            n_clusters=num_obs,
            max_iter=self.max_iter,
            n_init=self.num_initializations,
            init='random')
        var_dp_gmm.fit(observations)
        cluster_assignment_posteriors = var_dp_gmm.predict_proba(observations)
        cluster_assignment_posteriors_running_sum = np.cumsum(
            cluster_assignment_posteriors,
            axis=0)
        params = dict(means=var_dp_gmm.means_,
                      covs=var_dp_gmm.covariances_)

        total_mass_per_cluster = np.sum(cluster_assignment_posteriors, axis=0)
        num_inferred_clusters = np.sum(total_mass_per_cluster >= 1.)

        self.fit_results = dict(
            cluster_assignment_posteriors=cluster_assignment_posteriors,
            cluster_assignment_posteriors_running_sum=cluster_assignment_posteriors_running_sum,
            num_inferred_clusters=num_inferred_clusters,
            parameters=params,
        )

        return self.fit_results

    def centroids_after_last_obs(self) -> np.ndarray:
        """
        Returns array of shape (num features, feature dimension)
        """
        return self.fit_results['parameters']['means']
