import random
import numpy as np

random.seed(0)

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



def train_test_split(usr_photos, train_ratio, ratios):
    """
    Parameters
    ----------
        usr_photos: dict
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}
        
        train_ratio: float
            training data ratio

        ratios: list
            ratios for training mini-batches

    Results
    -------
        train_test_info: dict
            {train: {ratio: {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}, ...},
            test: {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...},
            users: [user1, user2, ...]}
            
    """

    train_test_info = {}
    total_train = {}
    test = {}
    valid_users = []

    N_users = len(list(usr_photos.keys()))
    N_train = int(N_users*train_ratio)


    users = list(usr_photos.keys())
    random.shuffle(users)
    count = 0
    for user in users:
        if user:
            count += 1
            if count <= N_train:
                total_train[user] = usr_photos[user]
            else:
                test[user] = usr_photos[user]

            valid_users.append(user)

    minibatches = train_minibatches(total_train, ratios)

    train_test_info['train'] = minibatches
    train_test_info['test'] = test
    train_test_info['ratio-minibacthes'] = ratios
    train_test_info['users'] = valid_users

    return train_test_info