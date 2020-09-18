import os
import numpy as np
from joblib import dump
from situ.acronym import load_acronym

def agg_features(com_features):
    """Aggregate exposure features of all images
        in the community.

    :param: com_features : dict
        photos and it features
           {user1: {photo1: [transformed features], ...}, ...}

    Returns
    -------
        transformed_features : list
            [ [user1's photo1's transformed features], ...]

    """
    features = []

    for user, user_features in com_features.items():
        for photo, photo_features in user_features.items():
            features.append(photo_features)
    features = np.asarray(features)

    return features


def train_clusteror(situ_name, model, com_features, cfg):
    """
    Train clusteror on all images of the community, which will be used
    further to cluster each user's image.

    Parameters
    ----------
    model: object
        clusteror model

    com_features : dict
        community exposure features
        dict of all users in a given situation with their clusteror features
            {user1: {photo1:[transformed features], ...}, ...}

    """
    aggfeatures = agg_features(com_features)
    ACRONYMS = load_acronym()

    if cfg.CLUSTEROR.TYPE == 'K_MEANS':
        model.fit(aggfeatures)

    save_path = os.path.join(cfg.OUTPUT.DIR, 'clusteror_'+ACRONYMS[situ_name]+'.joblib')
    if cfg.OUTPUT.VERBOSE:
        print('Save clusteror model !!! ')

    dump(model, save_path)

    return model