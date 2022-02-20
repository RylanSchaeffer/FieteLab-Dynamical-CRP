import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict

from rncrp.inference.base import BaseModel


def _init_centroids_dpmeans(X: np.ndarray,
                            max_distance_param: float):
    n_samples, n_features = X.shape

    # We can have (up to) as many clusters as samples
    chosen_center_indices = np.full(shape=n_samples,
                                    fill_value=False,
                                    dtype=np.bool)

    # Always take the first observation as a center, since no centroids exist
    # to compare against.
    chosen_center_indices[0] = True

    # Consequently, we start with the first observation
    for obs_idx in range(1, n_samples):
        x = X[obs_idx, np.newaxis, :]  # Shape: (1, sample dim)
        distances_x_to_centers = cdist(x, X[chosen_center_indices, :])
        if np.min(distances_x_to_centers) > max_distance_param:
            chosen_center_indices[obs_idx] = True

    chosen_centers = X[chosen_center_indices, :].copy()
    return chosen_centers


class DPMeans(BaseModel):

    def __init__(self,
                 gen_model_params: Dict[str, Dict[str, float]],
                 model_str: str = 'DP-Means',
                 plot_dir: str = None,
                 max_iter: int = 300,
                 num_initializations: int = 1,
                 **kwargs):

        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing_params']
        assert self.mixing_params['lambda'] > 0.
        self.component_prior_params = gen_model_params['component_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']

        # if max_iter = 1, then this is "online."
        # if max_iter > 1, then this if "offline"
        assert isinstance(max_iter, int)
        assert max_iter >= 1
        self.max_iter = max_iter

        # TODO: Currently unused
        self.num_initializations = num_initializations
        self.model_str = model_str
        self.plot_dir = plot_dir

        self.cluster_centers_ = None
        self.num_clusters_ = None
        self.num_init_clusters_ = None
        self.cluster_assignments_ = None
        self.n_iter_ = None
        self.loss_ = None

        self.fit_results = None

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray
            ):

        num_obs, obs_dim = observations.shape

        max_distance_param = self.mixing_params['lambda']

        centers_init = _init_centroids_dpmeans(
            X=observations,
            max_distance_param=max_distance_param)

        centers = np.zeros_like(observations)
        num_centers = centers_init.shape[0]
        centers[:num_centers, :] = centers_init

        cluster_assignments = np.full(num_obs, fill_value=-1, dtype=np.int)

        iter_idx = 0
        for iter_idx in range(self.max_iter):

            print(f'Num centers: {num_centers}')

            no_datum_reassigned = True

            # Assign data to centers.
            for obs_idx in range(num_obs):

                # Compute distance from datum to each centroids.
                distances = cdist(
                    XA=observations[obs_idx, np.newaxis, :],
                    XB=centers[:num_centers, :])

                # If smallest distance greater than cutoff max_distance_param,
                # then we create a new cluster.
                if np.min(distances) > max_distance_param:

                    # centroid of new cluster = new sample
                    centers[num_centers, :] = observations[obs_idx, :]
                    new_assigned_cluster = num_centers

                    # increment number of clusters by 1
                    num_centers += 1

                else:
                    # If the smallest distance is less than the cutoff max_distance_param, assign point
                    # to one of the older clusters
                    new_assigned_cluster = np.argmin(distances)

                # Check whether this datum is being assigned to a new center.
                if cluster_assignments[obs_idx] != new_assigned_cluster:
                    no_datum_reassigned = False

                # Record the observation's assignment
                cluster_assignments[obs_idx] = new_assigned_cluster

            # If no data was assigned to a different cluster, then we've converged.
            if iter_idx > 0 and no_datum_reassigned:
                break

            # Update centers based on assigned data.
            centers_to_keep = np.full(centers.shape[0],
                                      fill_value=False,
                                      dtype=np.bool)
            for center_idx in range(num_centers):

                # Get indices of all observations assigned to that cluster.
                indices_of_points_in_assigned_cluster = cluster_assignments == center_idx

                # Get observations assigned to that cluster.
                points_in_assigned_cluster = observations[indices_of_points_in_assigned_cluster, :]

                # If this cluster has no assigned points, skip.
                if points_in_assigned_cluster.shape[0] == 0:
                    continue

                # If this cluster has assigned points, mark to not delete.
                centers_to_keep[center_idx] = True

                # Recompute centroid from assigned observations.
                centers[center_idx, :] = np.mean(points_in_assigned_cluster,
                                                 axis=0)

            # Delete old clusters and shift remaining clusters up.
            centers_to_keep = centers[centers_to_keep, :]
            num_centers = centers_to_keep.shape[0]
            centers[:num_centers, :] = centers_to_keep
            centers[num_centers:, :] = 0.

        # Increment by 1 since range starts at 0 but humans start counting at 1
        iter_idx += 1

        # Clean up centers by removing any center with no data assigned.
        cluster_ids, num_assigned_points = np.unique(cluster_assignments,
                                                     return_counts=True)
        nonempty_clusters = cluster_ids[num_assigned_points > 0]
        centers = centers[nonempty_clusters]

        parameters = dict(means=centers)

        cluster_assignment_posteriors = np.eye(num_obs)[cluster_assignments]
        cluster_assignment_posteriors_running_sum = np.cumsum(cluster_assignment_posteriors,
                                                              axis=0)

        self.fit_results = dict(
            cluster_assignment_posteriors=cluster_assignment_posteriors,
            cluster_assignment_posteriors_running_sum=cluster_assignment_posteriors_running_sum,
            num_inferred_clusters=len(nonempty_clusters),
            parameters=parameters,
        )

        return self.fit_results

    def centroids_after_last_obs(self) -> np.ndarray:
        """
        Returns array of shape (num features, feature dimension)
        """
        return self.fit_results['parameters']['means']

    # def score(self, X, y=None, sample_weight=None):
    #     """Opposite of the value of X on the K-means objective.
    #
    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         New data.
    #
    #     y : Ignored
    #         Not used, present here for API consistency by convention.
    #
    #     sample_weight : array-like of shape (n_samples,), default=None
    #         The weights for each observation in X. If None, all observations
    #         are assigned equal weight.
    #
    #     Returns
    #     -------
    #     score : float
    #         Opposite of the value of X on the K-means objective.
    #     """
    #     check_is_fitted(self)
    #
    #     X = self._check_test_data(X)
    #     x_squared_norms = row_norms(X, squared=True)
    #     sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
    #
    #     return -_labels_inertia(X, sample_weight, x_squared_norms,
    #                             self.cluster_centers_)[1]
    #
    # def predict(self, X, sample_weight=None):
    #     """Predict the closest cluster each sample in X belongs to.
    #
    #     In the vector quantization literature, `cluster_centers_` is called
    #     the code book and each value returned by `predict` is the index of
    #     the closest code in the code book.
    #
    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         New data to predict.
    #
    #     sample_weight : array-like of shape (n_samples,), default=None
    #         The weights for each observation in X. If None, all observations
    #         are assigned equal weight.
    #
    #     Returns
    #     -------
    #     labels : ndarray of shape (n_samples,)
    #         Index of the cluster each sample belongs to.
    #     """
    #     check_is_fitted(self)
    #
    #     distances_x_to_centers = cdist(X, self.cluster_centers_)
    #
    #     # sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
    #
    #     distances_to_cluster_centers = euclidean_distances(
    #         X=X,
    #         Y=self.cluster_centers_,
    #         squared=False)
    #
    #     labels = np.argmin(distances_to_cluster_centers, axis=1)
    #
    #     return labels
