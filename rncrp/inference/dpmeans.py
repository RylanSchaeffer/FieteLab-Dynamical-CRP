import numpy as np
import scipy.spatial.distance
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict

from rncrp.inference.base import BaseModel


class DPMeans(BaseModel):

    def __init__(self,
                 model_str: str,
                 gen_model_params: Dict[str, Dict[str, float]],
                 plot_dir: str = None,
                 max_distance_param: float = 10.,
                 max_iter: int = 300,
                 num_initializations: int = 1,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: int = None,
                 copy_x: bool = True,
                 algorithm: str = 'auto',
                 **kwargs):

        self.gen_model_params = gen_model_params
        self.gen_model_params = gen_model_params
        self.mixing_params = gen_model_params['mixing']
        assert self.mixing_params['alpha'] > 0.
        assert self.mixing_params['beta'] == 0.
        self.feature_prior_params = gen_model_params['feature_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        # if max_iter = 1, then this is "online."
        # if max_iter > 1, then this if "offline"
        assert isinstance(max_iter, int)
        assert max_iter >= 1
        self.max_iter = max_iter
        self.num_initializations = num_initializations
        self.model_str = model_str
        self.plot_dir = plot_dir

        self.cluster_centers_ = None
        self.num_clusters_ = None
        self.num_init_clusters_ = None
        self.labels_ = None
        self.n_iter_ = None
        self.loss_ = None

    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray
            ):

        num_obs, obs_dim = observations.shape

        # Each datum might be its own cluster.
        max_num_clusters = num_obs

        centers = observations.copy()
        # We can have (up to) as many clusters as samples
        chosen_center_indices = np.zeros(shape=num_obs, dtype=np.bool)

        # Always take the first observation as a center, since no centroids exist
        # to compare against.
        chosen_center_indices[0] = True
        cluster_assignments = np.zeros(num_obs, dtype=np.int)

        for iter_idx in range(self.max_iter):

            no_datum_reassigned = True

            # Assign data to centers.
            for obs_idx in range(num_obs):

                # compute distance of new sample from previous centroids:
                distances = scipy.spatial.distance.cdist(
                    XA=X[obs_idx, np.newaxis, :],
                    XB=centers[chosen_center_indices, :])

                # If smallest distance greater than cutoff max_distance_param,
                # then we create a new cluster.
                if np.min(distances) > max_distance_param:

                    # centroid of new cluster = new sample
                    centers[num_centers, :] = X[obs_idx, :]
                    new_assigned_cluster = num_centers

                    # increment number of clusters by 1
                    num_centers += 1

                else:
                    # If the smallest distance is less than the cutoff max_distance_param, assign point
                    # to one of the older clusters
                    new_assigned_cluster = np.argmin(distances)

                # Check whether this datum is being assigned to a new center.
                if iter_idx > 0 and cluster_assignments[obs_idx] != new_assigned_cluster:
                    no_datum_reassigned = False

                # Record the observation's assignment
                cluster_assignments[obs_idx] = new_assigned_cluster

            # If no data was assigned to a different cluster, then we've converged.
            if iter_idx > 0 and no_datum_reassigned:
                break

            # Update centers based on assigned data.
            for center_idx in range(num_centers):

                # Get indices of all observations assigned to that cluster.
                indices_of_points_in_assigned_cluster = cluster_assignments == center_idx

                # Get observations assigned to that cluster.
                points_in_assigned_cluster = X[indices_of_points_in_assigned_cluster, :]

                if points_in_assigned_cluster.shape[0] >= 1:
                    # Recompute centroid from assigned observations.
                    centers[center_idx, :] = np.mean(points_in_assigned_cluster,
                                                     axis=0)

        # Increment by 1 since range starts at 0 but humans start counting at 1
        iter_idx += 1

        # Clean up centers by removing any center with no data assigned
        cluster_ids, num_assigned_points = np.unique(cluster_assignments,
                                                     return_counts=True)
        nonempty_clusters = cluster_ids[num_assigned_points > 0]
        centers = centers[nonempty_clusters]

        return cluster_assignments, centers, iter_idx

        # Initialize centers
        centers_init = self._init_centroids(
            X, x_squared_norms=x_squared_norms, init=init,
            random_state=random_state)
        self.num_init_clusters_ = centers_init.shape[0]


        # Shape: (num centers, num data)
        squared_distances_to_centers = euclidean_distances(
            X=centers,
            Y=X,
            squared=True)
        # Shape: (num data,)
        squared_distances_to_nearest_center = np.min(
            squared_distances_to_centers,
            axis=0)
        loss = np.sum(squared_distances_to_nearest_center)

        self.cluster_centers_ = centers
        self.num_clusters_ = centers.shape[0]
        self.labels_ = labels
        self.n_iter_ = n_iter_
        self.loss_ = loss
        return self

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return -_labels_inertia(X, sample_weight, x_squared_norms,
                                self.cluster_centers_)[1]

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        distances_x_to_centers = cdist(X, self.cluster_centers_)

        # sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        distances_to_cluster_centers = euclidean_distances(
            X=X,
            Y=self.cluster_centers_,
            squared=False)

        labels = np.argmin(distances_to_cluster_centers, axis=1)

        return labels
