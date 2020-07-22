def photo_expo(photo, f_top, detectors):
    
    """Estimate photo exposure
    
    Parameters
    ----------
    photo : dict
        objects in photo and its detection score
            {class1: [obj1, obj2,...], ... }

    f_top : float [0,1)
        A top N ranked detection object scores

    detectors : dict
        active ditector in a given situation and its score
            {detector1: score, ...}

    Returns
    -------
        photo_expo : float

    """
    photo_expo = 0
    sum_objectness = 0


    for object_, scores in photo.items():
        
        if object_ in detectors:
            
            objectness = sum([score for score in scores if score >= f_top])
            
            sum_objectness += objectness
            photo_expo += objectness*detectors[object_]

    photo_expo = photo_expo/objectness

    return photo_expo