import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from typing import Dict

from rncrp.inference.base import BaseModel


class KMeansWrapper(BaseModel):
    """
    KMeans. Wrapper around scikit-learn's implementation.
    """

    def __init__(self,
                 n_clusters: int,
                 model_str: str = 'VI-GMM',
                 plot_dir: str = None,
                 max_iter: int = 100,
                 num_initializations: int = 10,
                 **kwargs,
                 ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_initializations = num_initializations
        self.model_str = model_str
        self.plot_dir = plot_dir
        self.fit_results = None

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray):

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.num_initializations,
            init='random')
        kmeans.fit(observations)
        assigned_clusters = kmeans.predict(observations)
        cluster_assignment_posteriors = OneHotEncoder(sparse=False).fit_transform(
            assigned_clusters.reshape(-1, 1))

        # Reorder to "Left Ordered Form".
        shuffle_indices = np.lexsort(-cluster_assignment_posteriors[::-1])
        cluster_assignment_posteriors = cluster_assignment_posteriors[:, shuffle_indices]

        cluster_assignment_posteriors_running_sum = np.cumsum(
            cluster_assignment_posteriors,
            axis=0)
        params = dict(means=kmeans.cluster_centers_[shuffle_indices, :])

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
