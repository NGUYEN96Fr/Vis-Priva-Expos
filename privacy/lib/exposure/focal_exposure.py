import math

def sigmoid_scaling(x):

    return 6*(1/(1+math.exp(-x)) - 0.5)


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
    scaled_expo =  (1/(1-(1/K)*abs(expo))**gamma)*expo
    return sigmoid_scaling(scaled_expo)