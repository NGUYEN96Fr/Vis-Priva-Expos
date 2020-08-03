import numpy as np
from photo_situ_expos.photo_expo import photo_expo

def _photos_user(user_photos, f_top, detectors, opt_threshs, filter):
    """Estimate photo exposure for an user

    Parameters
    ----------
        user_photos : dict
            user photos and objectness scores
                {photo1: {class1: [obj1, ...], ...},...}

        f_top : float [0,1)
            A top N ranked detection object scores

        detectors : dict
            active detectors for a given situation
        
    Returns
    -------
        photo_expo_per_user : dict
            photo exposure and its objectness
                {photo1: ( expo +, expo -, sum_objectness),...}

    """
    photos_expo_per_user = {}

    for photo in user_photos:
        expo_pos, expo_neg, sum_objectness =  photo_expo(user_photos[photo], f_top, detectors, opt_threshs)
        if filter:
            if abs(expo_pos) + abs(expo_neg) >= 0.01:
                photos_expo_per_user[photo] = photo_expo(user_photos[photo], f_top, detectors, opt_threshs)

    return photos_expo_per_user


def usr_photo_expo(users, f_top, detectors, opt_threshs, filter = False):
    """Estimate photo exposure for all users in a given situation

    Parameters
    ----------
        users : dict
            users and their photos
                {user1: {photo1: {class1: [obj1, ...], ...}, ...}, ...}
        
        f_top : float [0,1)
            A top N ranked detection object scores

        detectors : dict
            active detectors for a given situation

        opt_threshs : dict
            optimal active detector thresholds
                {detector1: thresh1, ...}

        filter : boolean
            if filtering neutral images

    Returns
    -------
        photo_user_expos : dict
            {user1: {photo1: (expo +, expo-, sum_objectness), ...}, ...}

    """
    expo = {}

    for user, photos in users.items():
        photo_expos = _photos_user(photos, f_top, detectors, opt_threshs, filter)

        if len(list(photo_expos.keys())) >= 4:
            expo[user] = photo_expos

    return expo


def top_pos_neg_photos(user_expo):
    """
    Extract top 20 % positive and negative photos of a user profile

    :param user_expo:
        {photo1: (expo +, expo-, sum_objectness), ...}

    :return:
    """
    photos = list(user_expo.keys())
    n_photos = int(0.2*len(photos))
    if n_photos <= 5:
        n_photos = len(photos)

    neg_scores = [user_expo[user][1] for user in list(user_expo.keys())]
    neg_sort_max_to_min = np.argsort(np.asarray(neg_scores))[::-1][:n_photos]
    pos_scores = [user_expo[user][0] for user in list(user_expo.keys())]
    pos_sort_max_to_min = np.argsort(np.asarray(pos_scores))[::-1][:n_photos]