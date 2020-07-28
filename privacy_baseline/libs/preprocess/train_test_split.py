def train_minibatches(training_data, ratios):
    """Split training data into many training mini batches

    Parameters
    ----------
        training_data : dict
            containing user photos
                {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}
        
        ratios: list of ratios
            [10, 30, 50 ...,100]

    Results
    -------
        minibatches : dict
            {ratio: {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}, ...}

    """
    minibatches = {}
    N_users =  len(list(training_data.keys())) #nb of users
    
    for ratio in ratios:
        minibatches[ratio] = {}
        nb_img_ratio = int(ratio/100*N_users)
        k = 0
        for user, photos in training_data.items():
            k +=1
            if k <= nb_img_ratio:
                minibatches[ratio][user] = photos
            else:
                break

    return minibatches


def train_test_split(usr_photos, train_ratio):
    """
    Parameters
    ----------
        usr_photos: dict
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}
        
        train_ratio: float
            training data ratio

    Results
    -------

        save files to
            out/train_test/
            
    """