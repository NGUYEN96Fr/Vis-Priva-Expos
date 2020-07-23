from focal_exposure.focal_exposure import focal_exposure


def feature_photo(photo_expo, gamma, K= 10):
    """Extracting features from a photo

    :param photo_expo: tuple
        initial photo exposure and its objectness sum
            (exposure,sum_objectness)

    :param gamma: float
        focusing factor

    :param  K : float
        rescaling constant

    :return: features: tuple
        tuple of photo features
            (f_expo,f_obj)

    """
    initial_expo, objness = photo_expo
    scaled_expo = focal_exposure(initial_expo, gamma, K)

    f_expo = scaled_expo
    f_obj = objness
    features = (f_expo, f_obj)

    return features


def feature_user_photos(expo_photos, gamma, K= 10):
    """Extracting features for all user's photos
    :param: expo_photos: dict
        exposure of all user's photo and its objectness scores
            {photo1: (expo, sum_objectness), ...}
    :param gamma:

    :param K:

    :return: user_photo_features : dict
            {photo1: (f_expo, f_obj), ...}

    """
    user_photo_features ={}
    for photo_name, expo_ in expo_photos.items():
        user_photo_features[photo_name] = feature_photo(expo_, gamma, K)

    return user_photo_features