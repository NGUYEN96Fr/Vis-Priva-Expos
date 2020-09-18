from focal_exposure.focal_exposure import focal_exposure

def feature_transform(f_expo_pos, f_expo_neg, f_dens, transform):
    """
    Apply feature transform on photo features scaled by focal exposure

    :param f_expo_pos:
    :param f_expo_neg:
    :param f_dens:
    :param transform: transforming method
    :return:
        transformed features

    """

    if transform == 'ABS':
        f_abs = abs(f_expo_pos) + abs(f_expo_neg)
        return [f_abs, f_dens]

    if transform == 'ORG':
        return [f_expo_pos, f_expo_neg, f_dens]

    if transform == 'POS_NEG':
        return [f_expo_pos, f_expo_neg]


def photo_features(photo_expo, gamma, K= 10, transform = 'ORG'):
    """Extracting features from a photo

    :param photo_expo: tuple
        initial photo exposure and its objectness sum
            (expo +, expo -, sum_objectness)


    :param gamma: float
        focusing factor

    :param  K : float
        rescaling constant

    :return: features: tuple
        tuple of photo features

    """
    expo_pos, expo_neg , objness = photo_expo
    f_expo_pos = focal_exposure(expo_pos, gamma, K)
    f_expo_neg = focal_exposure(expo_neg, gamma, K)
    f_obj = objness

    # apply feature transform
    features = feature_transform(f_expo_pos, f_expo_neg, f_obj, transform)

    return features


def user_features(expo_photos, gamma, K= 10, transform = 'ORG'):
    """Extracting features for all user's photos
    :param: expo_photos: dict
        exposure of all user's photo and its objectness corr
            {photo1: (expo +, expo -, sum_objectness), ...}
    :param gamma:

    :param K:

    :return: features : dict
            {photo1: [], ...}

    """
    features ={}
    for photo_name, expo_ in expo_photos.items():
        features[photo_name] = photo_features(expo_, gamma, K, transform)

    return features


def community_features(users, gamma, K,  transform):
    """Calculate exposure features of all users in a given situation
        The features will be used for the clusteror process.

    :param: users: dict
        users, their photos, and photo exposure and its objectness
        {user1:{photo1:(expo +, expo -, obj),...},...}

    :param: gamma: focal factor
    :param: transform: string
            transforming method
    :param: K: scaling factor

    Returns
    -------
    clustering_features: dict
        {user1: {photo1:[transformed features], ...}, ...}

    """
    features = {}

    for user, expo_photos in users.items():
        features[user] = user_features(expo_photos, gamma, K, transform)

    return features