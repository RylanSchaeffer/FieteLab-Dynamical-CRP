import numpy as np
import sklearn.metrics
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score, f1_score
import sklearn.metrics.cluster
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict, Tuple


def purity_score(labels_true: np.ndarray,
                 labels_pred: np.ndarray):
    """
    Purity: For each cluster, assign it the cluster label that is most prevalent amongst
    points assinged to the cluster. Purity is the number of correctly matched class
    and cluster labels, divided by the number of total data points

    See: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Implementation copied from https://stackoverflow.com/a/51672699/4570472
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def compute_predicted_clusters_scores(cluster_assignment_posteriors: np.ndarray,
                                      true_cluster_assignments: np.ndarray,
                                      ) -> Tuple[Dict[str, float], np.ndarray]:
    """

    Parameters:
        cluster_assignment_posteriors: shape (number obs, num clusters).
            (r, c)th element is probability the rth observation belongs to cth cluster
        true_cluster_assignments: integer classes with shape (num obs, )
            or binary with shape (num obs, max num clusters)
    """

    if len(cluster_assignment_posteriors.shape) == 2:
        pred_cluster_labels = np.argmax(cluster_assignment_posteriors,
                                        axis=1)
    elif len(cluster_assignment_posteriors.shape) == 1:
        pred_cluster_labels = cluster_assignment_posteriors
    else:
        raise ValueError('Wrong number of dimensions.')

    rnd_score = rand_score(labels_pred=pred_cluster_labels,
                           labels_true=true_cluster_assignments)

    adj_rnd_score = adjusted_rand_score(labels_pred=pred_cluster_labels,
                                        labels_true=true_cluster_assignments)

    adj_mut_inf_score = adjusted_mutual_info_score(labels_pred=pred_cluster_labels,
                                                   labels_true=true_cluster_assignments)

    norm_mut_inf_score = normalized_mutual_info_score(labels_pred=pred_cluster_labels,
                                                      labels_true=true_cluster_assignments)

    pur_score = purity_score(labels_true=true_cluster_assignments,
                             labels_pred=pred_cluster_labels)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    f_score = f1_score(
        true_cluster_assignments,
        pred_cluster_labels,
        average='macro')

    scores_results = {
        'Rand Score': rnd_score,
        'Adjusted Rand Score': adj_rnd_score,
        'Adjusted Mutual Info Score': adj_mut_inf_score,
        'Normalized Mutual Info Score': norm_mut_inf_score,
        'Purity Score': pur_score,
        'F Score (Macro)': f_score,
    }

    return scores_results, pred_cluster_labels


def compute_sum_of_squared_distances_to_nearest_center(X: np.ndarray,
                                                       centroids: np.ndarray,
                                                       ) -> float:
    """
    Args:
        X: (num data, obs dim)
        centroids: (num centroids, obs dim)
    """

    # Shape: (num centroids, num data)
    squared_distances_to_centers = euclidean_distances(
        X=centroids,
        Y=X,
        squared=True)
    # Shape: (num data,)
    squared_distances_to_nearest_center = np.min(
        squared_distances_to_centers,
        axis=0)
    sum_of_squared_distances_to_nearest_center = np.sum(squared_distances_to_nearest_center)

    return sum_of_squared_distances_to_nearest_center


def compute_cluster_linear_regression_score(cluster_assignment_posteriors: np.ndarray,
                                            targets: np.ndarray,
                                            fit_intercept: bool = True,
                                            use_vanilla: bool = True,
                                            use_ridge: bool = False,
                                            use_lasso: bool = False):

    assert cluster_assignment_posteriors.shape[0] == targets.shape[0]
    assert use_vanilla + use_lasso + use_ridge == 1

    if use_vanilla:
        from sklearn.linear_model import LinearRegression
        linear_model = LinearRegression(fit_intercept=fit_intercept)
    elif use_ridge:
        from sklearn.linear_model import Ridge
        linear_model = Ridge(fit_intercept=fit_intercept)
    elif use_lasso:
        from sklearn.linear_model import Lasso
        linear_model = Lasso(fit_intercept=fit_intercept)
    else:
        raise ValueError('How did you end up here?')

    linear_model.fit(X=cluster_assignment_posteriors, y=targets)
    predicted_targets = linear_model.predict(cluster_assignment_posteriors)

    # R^2: 1 - Sum of Squared Residuals / Total sum of squares
    coeff_of_determination = linear_model.score(X=cluster_assignment_posteriors, y=targets)

    cluster_linear_regression_results = {
        'coeff_of_determination': coeff_of_determination,
        'predicted_targets': predicted_targets,
    }

    return cluster_linear_regression_results


def compute_cluster_multiclass_classification_score(cluster_assignment_posteriors: np.ndarray,
                                                    targets: np.ndarray,
                                                    fit_intercept: bool = True,
                                                    use_vanilla: bool = True,
                                                    use_ridge: bool = False,
                                                    use_lasso: bool = False):

    assert cluster_assignment_posteriors.shape[0] == targets.shape[0]
    assert use_vanilla + use_lasso + use_ridge == 1

    from sklearn.neural_network import MLPClassifier

    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(),  # No hidden layer
    )
    mlp_classifier.fit(X=cluster_assignment_posteriors, y=targets)
    predicted_targets = mlp_classifier.predict(cluster_assignment_posteriors)

    avg_acc = mlp_classifier.score(X=cluster_assignment_posteriors, y=targets)

    cluster_linear_regression_results = {
        'avg_acc': avg_acc,
        'predicted_targets': predicted_targets,
    }

    return cluster_linear_regression_results
