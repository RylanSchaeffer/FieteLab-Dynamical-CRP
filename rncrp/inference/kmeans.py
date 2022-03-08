import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder
from typing import Dict

from rncrp.inference.base import BaseModel


class KMeans(BaseModel):
    """
    KMeans. Wrapper around scikit-learn's implementation.
    """

    def __init__(self,
                 n_clusters: int,
                 gen_model_params: Dict[str, Dict[str, float]],
                 model_str: str = None,
                 plot_dir: str = None,
                 max_iter: int = 100,
                 # num_initializations: int = 10,
                 **kwargs,
                 ):

        if model_str is None:
            if max_iter == 1:
                model_str = 'K-Means (Online)'
            else:
                model_str = 'K-Means (Offline)'

        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.component_prior_params = gen_model_params['component_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        # self.num_initializations = num_initializations
        self.model_str = model_str
        self.plot_dir = plot_dir
        self.fit_results = None

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray):

        num_obs, obs_dim = observations.shape

        # If "online", we initialize centroids randomly from prior on centroids.
        if self.max_iter == 1:
            centers = np.random.multivariate_normal(
                mean=np.zeros(obs_dim),
                cov=self.component_prior_params['centroids_prior_cov_prefactor'] * np.eye(observations.shape[1]),
                size=self.n_clusters,)
        # If "offline", we initialize centroids using data.
        else:
            centers = np.random.choice(
                observations,
                size=self.n_clusters,
                replace=False,
            )

        cluster_assignments_posteriors = np.full(num_obs, fill_value=-1, dtype=np.int)

        for iter_idx in range(self.max_iter):

            # Set flag to know whether we have converged.
            datum_reassigned = False

            # Assign observations to clusters.
            for obs_idx in range(num_obs):

                # Assign points to  nearest centroids.
                distances_to_centers = cdist(
                    XA=observations[obs_idx, np.newaxis, :],
                    XB=centers)

                new_assigned_cluster = np.argmin(distances_to_centers)

                # Check whether this datum is being assigned to a new center.
                if cluster_assignments_posteriors[obs_idx] != new_assigned_cluster:
                    datum_reassigned = True

                # Record the observation's assignment
                cluster_assignments_posteriors[obs_idx] = new_assigned_cluster

            # If no data was assigned to a different cluster, then we've converged.
            if iter_idx > 0 and not datum_reassigned:
                break

            # Update centers based on assigned observations.
            for center_idx in range(self.n_clusters):

                # Get indices of all observations assigned to that cluster.
                indices_of_points_in_assigned_cluster = cluster_assignments_posteriors == center_idx

                # Get observations assigned to that cluster.
                points_in_assigned_cluster = observations[indices_of_points_in_assigned_cluster, :]

                # If this cluster has no assigned points, skip.
                if points_in_assigned_cluster.shape[0] == 0:
                    continue

                # Recompute centroid from assigned observations.
                centers[center_idx, :] = np.mean(points_in_assigned_cluster, axis=0)

        cluster_assignment_posteriors_one_hot = OneHotEncoder(sparse=False).fit_transform(
            cluster_assignments_posteriors.reshape(-1, 1))

        # Reorder to "Left Ordered Form".
        shuffle_indices = np.lexsort(-cluster_assignment_posteriors_one_hot[::-1])
        cluster_assignment_posteriors_one_hot = cluster_assignment_posteriors_one_hot[:, shuffle_indices]

        cluster_assignment_posteriors_running_sum = np.cumsum(
            cluster_assignment_posteriors_one_hot,
            axis=0)
        params = dict(means=centers[shuffle_indices, :])

        total_mass_per_cluster = np.sum(cluster_assignment_posteriors_one_hot, axis=0)
        num_inferred_clusters = np.sum(total_mass_per_cluster >= 1.)

        self.fit_results = dict(
            cluster_assignment_posteriors=cluster_assignment_posteriors_one_hot,
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
