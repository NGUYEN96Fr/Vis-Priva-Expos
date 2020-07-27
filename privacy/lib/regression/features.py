def user_feature(user_clusters):
    """
    :params: list of dicts
        the user's clusters, its components, and variance
            [{'centroid': , 'components': ,'variance':  },...]


    Returns
    -------
        regress_feature :list
            user regression features
                [feature1, ...]

    """
    regress_features = []
    for cluster in user_clusters:

        centroid = list(cluster['centroid'])
        for k in range(len(centroid)):
            regress_features.append(centroid[k])

        variance = cluster['variance']
        regress_features.append(variance)

    return regress_features


def regression_features(users):
    """Estimate regression features for all users

    :param: users: dict
        users and their clusters
        {user1: [{'centroid': , 'components': ,'variance':  },...], ...}

    Returns
    -------
        regression_feature_users : dict
            {user1: [feature1,...], ...}

    """
    regression_feature_users = {}
    for user, clusters in users.items():
        regression_feature_users[user] = user_feature(clusters)

    return regression_feature_users
