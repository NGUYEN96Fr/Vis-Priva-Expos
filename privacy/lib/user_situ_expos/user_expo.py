from photo_situ_expos.photo_expo import photo_expo

def _photos_user(user_photos, f_top, detectors):
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
        photos_expo_per_user[photo] = photo_expo(user_photos[photo], f_top, detectors)

    return photos_expo_per_user


def usr_photo_expo(users, f_top, detectors):
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

    Returns
    -------
        photo_user_expos : dict
            {user1: {photo1: (expo +, expo-, sum_objectness), ...}, ...}

    """
    expo = {}

    for user, photos in users.items():
        expo[user] = _photos_user(photos, f_top, detectors)

    return expo









