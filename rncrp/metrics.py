import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score
from typing import Dict, Tuple


def score_predicted_clusters(table_assignment_posteriors: np.ndarray,
                             true_cluster_labels: np.ndarray,
                             ) -> Tuple[Dict[str, float], np.ndarray]:

    # cluster assignment posteriors has shape (number obs, num clusters)
    # (r, c)th element is probability the rth observation belongs to cth cluster
    # true_cluster_labels: integer classes with shape (num obs, )
    
    pred_cluster_labels = np.argmax(table_assignment_posteriors,
                                    axis=1)

    rnd_score = rand_score(labels_pred=pred_cluster_labels,
                           labels_true=true_cluster_labels)

    adj_rnd_score = adjusted_rand_score(labels_pred=pred_cluster_labels,
                                        labels_true=true_cluster_labels)

    adj_mut_inf_score = adjusted_mutual_info_score(labels_pred=pred_cluster_labels,
                                                   labels_true=true_cluster_labels)

    norm_mut_inf_score = normalized_mutual_info_score(labels_pred=pred_cluster_labels,
                                                      labels_true=true_cluster_labels)

    scores_results = {
        'Rand Score': rnd_score,
        'Adjusted Rand Score': adj_rnd_score,
        'Adjusted Mutual Info Score': adj_mut_inf_score,
        'Normalized Mutual Info Score': norm_mut_inf_score,
    }

    return scores_results, pred_cluster_labels