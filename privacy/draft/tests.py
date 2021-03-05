
import numpy as np
import matplotlib.pyplot as plt

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

# expo = 0.8
# gammas = [1, 3, 5]
# objs = np.linspace(0,1,10)
#
# print('Original Expo: ',expo)
# for gamma in gammas:
#     print('########################')
#     print('gamma= ',gamma)
#     for obj in objs:
#         print('obj= ',obj,' rexpo= ',focal_concept(expo, obj, gamma,K = 10))

def sigmoid():
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


    X = np.linspace(-6,6,20)
    Y = (np.divide(1,1 + np.exp(-X)) -0.5)*6
    plt.plot(X,Y)
    plt.grid()
    fig = plt.gcf()
    fig.savefig('sigmoid.jpg')


def plot_score_avg():
    """"""

    RCNNs = [0.44, 0.52, 0.51, 0.56]
    MOBIs = [0.44, 0.45, 0.47, 0.51]
    Nb= ['100', '200', '300', '400']

# sigmoid()
