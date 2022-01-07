from rncrp.inference.base import BaseModel


class DPMeans(BaseModel):

    def __init__(self,
                 max_distance_param: float = 10.,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: int = None,
                 copy_x: bool = True,
                 algorithm: str = 'auto'):

        self.max_distance_param = max_distance_param
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

        self.cluster_centers_ = None
        self.num_clusters_ = None
        self.num_init_clusters_ = None
        self.labels_ = None
        self.n_iter_ = None
        self.loss_ = None

    def _init_centroids(self, X, x_squared_norms, init, random_state,
                        init_size=None):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        max_distance_param = self.max_distance_param

        # Randomly permute data
        X = X[random_state.permutation(n_samples)]

        if isinstance(init, str) and init == 'dp-means':
            centers = _init_centroids_dpmeans(X,
                                              max_distance_param=max_distance_param,
                                              x_squared_norms=x_squared_norms,
                                              random_state=random_state,)
        elif isinstance(init, str) and init == 'dp-means++':
            # centers = _init_centroids_dpmeans_plusplus_old(X, max_distance_param=max_distance_param,
            #                                            random_state=random_state,
            #                                            x_squared_norms=x_squared_norms)
            centers = _init_centroids_dpmeans_plusplus(X=X,
                                                       max_distance_param=max_distance_param,
                                                       x_squared_norms=x_squared_norms,
                                                       random_state=random_state)
        elif isinstance(init, str) and init == 'random':
            # seeds = random_state.permutation(n_samples)[:n_clusters]
            # centers = X[seeds]
            raise NotImplementedError
        else:
            raise ValueError(f"Init {init} must be one of: dp-means, dp-means++, random.")

        return centers

    def fit(self, X, y=None, sample_weight=None):
        random_state = check_random_state(self.random_state)
        # sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Validate init array
        init = self.init

        # subtract of mean of x for more accurate distance computations
        X = X.copy()
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, '__array__'):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self.algorithm in {"auto", "full"}:
            dpmeans_single = dp_means
            # self._check_mkl_vcomp(X, X.shape[0])
        else:
            raise NotImplementedError

        # Initialize centers
        centers_init = self._init_centroids(
            X, x_squared_norms=x_squared_norms, init=init,
            random_state=random_state)
        self.num_init_clusters_ = centers_init.shape[0]

        if self.verbose:
            print(f"Init method: {self.init} selected {self.num_init_clusters_}"
                  f" inital clusters with lambda={self.max_distance_param}.")

        # run DP-means
        labels, centers, n_iter_ = dpmeans_single(
            X=X, max_distance_param=self.max_distance_param,
            centers_init=centers_init, max_iter=self.max_iter,
        )

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            centers += X_mean

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
