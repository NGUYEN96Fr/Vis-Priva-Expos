
def retrieve_photos(path):
    """Retrieve photos per user

    Parameters
    ----------
        path : string
            path to user images (.txt file)

    
    Returns
    -------
        photos_per_user : dict
            images per user with initial photo exposures
                {user1:{photo1:None, ...}, ...}

    """
    photos_per_user = {}
    count_photos = {}

    with open(path) as fp:
        lines = fp.readlines()

        for line in lines:
            parts = line.split(' ')

            user_ID = parts[0]
            photo_ID = parts[1]

            if user_ID not in photos_per_user:
                photos_per_user[user_ID] = {}
                count_photos[user_ID] = 0
            
            if photo_ID  not in photos_per_user[user_ID]:
                photos_per_user[user_ID][photo_ID] = None
                count_photos[user_ID] += 1

    print('Number of user: ',len(photos_per_user))
    print('Number of photos per user: ',count_photos)

    return photos_per_user



def retrieve_detected_objects(path):
    """Retrieve detected objects per photo per user


    Parameters
    ----------
        path : string
            path to user images (.txt file)

    
    Returns
    -------
        objects_photo_per_user : dict
            detected objects per photo per user
                {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}
    """

    objects_photo_per_user = {}

    with open(path) as fp:
        lines = fp.readlines()
        
        for line in lines:
            parts = line.split(' ')

            userID = parts[0]
            photoID = parts[1]
            class_ = parts[3]
            objectness = float(parts[4])

            if userID not in objects_photo_per_user:
                objects_photo_per_user[userID] = {}
            
            if photoID not in objects_photo_per_user[userID]:
                objects_photo_per_user[userID][photoID] = {}
            
            if class_ not in objects_photo_per_user[userID][photoID]:
                objects_photo_per_user[userID][photoID][class_] = []

            objects_photo_per_user[userID][photoID][class_].append(objectness)

    return objects_photo_per_user