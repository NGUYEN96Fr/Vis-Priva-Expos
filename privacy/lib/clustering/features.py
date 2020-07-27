import numpy as np
import matplotlib.pyplot as plt
from focal_exposure.focal_exposure import focal_exposure


def feature_photo(photo_expo, gamma, K= 10):
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
            (f_expo + , f_exp -, f_obj)

    """
    expo_pos, expo_neg , objness = photo_expo
    f_expo_pos = focal_exposure(expo_pos, gamma, K)
    f_expo_neg = focal_exposure(expo_neg, gamma, K)
    f_obj = objness

    features = [f_expo_pos, f_expo_neg,  f_obj]

    return features


def feature_user_photos(expo_photos, gamma, K= 10):
    """Extracting features for all user's photos
    :param: expo_photos: dict
        exposure of all user's photo and its objectness scores
            {photo1: (expo +, expo -, sum_objectness), ...}
    :param gamma:

    :param K:

    :return: user_photo_features : dict
            {photo1: (f_expo +, f_expo -, f_obj), ...}

    """
    user_photo_features ={}
    for photo_name, expo_ in expo_photos.items():
        user_photo_features[photo_name] = feature_photo(expo_, gamma, K)

    return user_photo_features


def clustering_photo_feature(situ_name , users, gamma, K=10, plot = False):
    """Calculate feature clustering of all users in a given situation

    :param: users: dict
        users, their photos, and photo exposure and its objectness
        {user1:{photo1:(expo +, expo -, obj),...},...}

    :param: gamma :

    :param: K

    Returns
    -------
    clustering_features: dict
        {user1: {photo1:(f_expo +, f_expo -, f_dense), ...}, ...}

    """
    clustering_features = {}
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for user, expo_photos in users.items():
        photo_features = feature_user_photos(expo_photos, gamma, K)
        if plot:

            f_expo_pos = []
            f_expo_neg = []
            f_dens = []
            for photo, features in photo_features.items():
                f_expo_pos.append(features[0])
                f_expo_neg.append(features[1])
                f_dens.append(features[2])
            ax.scatter(np.asarray(f_expo_pos), np.asarray(f_expo_neg), np.asarray(f_dens), marker=".")

        clustering_features[user] = photo_features

    if plot:
        plt.title(situ_name)
        # plt.ylabel('f_expo_neg')
        # plt.xlabel('f_expo_pos')
        # plt.zlabel('f_dens')

        ax.set_xlabel('f_expo_pos')
        ax.set_ylabel('f_expo_neg')
        ax.set_zlabel('f_dens')

        plt.savefig(situ_name + '_features.png')
        plt.clf()

    return clustering_features
