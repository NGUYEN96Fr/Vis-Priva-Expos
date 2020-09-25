
def focal_exposure(expo, gamma, K = 10):
    """Rescale  photo exposures

    Parameters
    ----------
        expo : float
            orginial photo exposure
        
        gamma : int
            focusing factor

        K : float
            rescaling constant

    Returns
    -------
        rescaled photo exposure
    """
    return (1/(1-(1/K)*abs(expo))**gamma)*expo


def focal_concept(expo, obj, gamma, K=10):
    """Focal concept exposures

    Parameters
    ----------
        expo : float
            orginal visual concept exposure

        obj: float
            object-ness detection

        gamma : int
            focusing factor

        K : float
            rescaling constant

    Returns
    -------
        rescaled photo exposure
    """
    scaled_expo =  (1*(1 - (1 / K) * abs(expo)) ** gamma) * expo
    #print(expo,'-->',scaled_expo)
    return scaled_expo