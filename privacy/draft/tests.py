
import numpy as np


def focal_concept(expo, obj, gamma, K=20):
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
    scaled_expo =  (1/(1 - (1 / K) * abs(expo)) ** gamma) * expo
    return scaled_expo

expo = 0.8
gammas = [1, 3, 5]
objs = np.linspace(0,1,10)

print('Original Expo: ',expo)
for gamma in gammas:
    print('########################')
    print('gamma= ',gamma)
    for obj in objs:
        print('obj= ',obj,' rexpo= ',focal_concept(expo, obj, gamma,K = 10))