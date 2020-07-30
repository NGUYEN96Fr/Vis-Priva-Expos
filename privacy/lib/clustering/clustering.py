from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as LA
from clustering.feature_transform import transform

def aggregate_features(user, feature_transform='origin'):
    """Transforming and Aggregating features

    :param: user : dict
        photos and it features
            {photo1: (f_expo +, f_expo -, f_dens), ...}

    :param: feature_transform: string
        feature transform method:
            + abs

    Returns
    -------
        aggregated_features : list
            [ [transformed feature1, transformed feature2, ...], ...]

    """
    aggregated_transformed_features = []
    for photo, features in user.items():
        transformed_features = transform(features[0], features[1], features[2], feature_transform)
        aggregated_transformed_features.append(transformed_features)

    return aggregated_transformed_features


def photo_expo_clustering(user_photo_features, N, feature_transform):
    """Clustering exposure of user's photos

    :param: user_photo_features
        clustering features extracted from user photos
            {photo1: (f_expo +, f_expo -, f_dens), ...}

    :param: N : int
        number of cluster

    :return:
        Groups: list of dict
            N clusters with its centroid, variance and its
                [{'centroid': , 'components': ,'variance':  },...]
    """
    aggregated_features = np.asarray(aggregate_features(user_photo_features, feature_transform))
    kmeans = KMeans(n_clusters=N, random_state=0).fit(aggregated_features)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    Groups = [{'centroid': centroids[i, :], 'components': None, 'variance': None} for i in range(N)]

    for k in range(N):
        Groups[k]['components'] = aggregated_features[np.where(labels == k)[0], :]
        Groups[k]['variance'] = LA.norm(aggregated_features[np.where(labels == k)[0], :], 'fro')

    return Groups


def photo_user_expo_clustering(clustering_feature_users, N, feature_transform):
    """

    Parameters
    ----------
    clustering_feature_users : dict
        list of all users in a given situation with their clustering features
            {user1: {photo1:(f_expo +, f_expo -, f_dense), ...}, ...}

    N : int
        number of clusters

    feature_transform: string
        apply feature transform on photo features

    Returns
    -------
        user_photo_grouping : dict
            grouping all photo's users
                {user1: [{'centroid': , 'components': ,'variance':  },...], ...}

    """
    user_photo_grouping = {}
    for user, clustering_feature in clustering_feature_users.items():
        user_photo_grouping[user] = photo_expo_clustering(clustering_feature, N, feature_transform)

    return user_photo_grouping