from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as LA

def aggregate_features(user):
    """Aggregating features

    :param: user : dict
        photos and it features
            {photo1: (scaled_f_expo +, scaled_f_expo -, f_dens), ...}

    Returns
    -------
        aggregated_features : list
            [ [scaled_f_expo +, scaled_f_expo -, f_dens], ...]

    """
    aggregated_features = []
    for photo, features in user.items():
        aggregated_features.append(features)

    return aggregated_features


def photo_expo_clustering(user_photo_features, N):
    """Clustering exposure of user's photos

    :param: user_photo_features
        clustering features extracted from user photos
            {photo1: (scaled_f_expo +, scaled_f_expo -, f_dens), ...}

    :param: N : int
        number of cluster

    :return:
        Groups: list of dict
            N clusters with its centroid, variance and its
                [{'centroid': , 'components': ,'variance':  },...]
    """
    aggregated_features = np.asarray(aggregate_features(user_photo_features))
    kmeans = KMeans(n_clusters=N, random_state=0).fit(aggregated_features)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    Groups = [{'centroid': centroids[i, :], 'components': None, 'variance': None} for i in range(N)]

    for k in range(N):
        Groups[k]['components'] = aggregated_features[np.where(labels == k)[0], :]
        Groups[k]['variance'] = LA.norm(aggregated_features[np.where(labels == k)[0], :], 'fro')

    return Groups


def photo_user_expo_clustering(clustering_feature_users, N = 4):
    """

    Parameters
    ----------
    clustering_feature_users : dict
        list of all users in a given situation with their clustering features
            {user1: {photo1:(f_expo +, f_expo -, f_dense), ...}, ...}

    N : int
        number of clusters

    Returns
    -------
        user_photo_grouping : dict
            grouping all photo's users
                {user1: [{'centroid': , 'components': ,'variance':  },...], ...}

    """
    user_photo_grouping = {}
    for user, clustering_feature in clustering_feature_users.items():
        user_photo_grouping[user] = photo_expo_clustering(clustering_feature, N)

    return user_photo_grouping