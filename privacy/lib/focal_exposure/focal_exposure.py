
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

    return (1/(1-1/K*abs(expo))**gamma)*expo