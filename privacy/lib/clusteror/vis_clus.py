"""
Visual concept object-ness detection clustering

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation as AF
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

def vis_clus(situ_name, x_community, detectors):
    """
    Returns
    -------

    """
    features = []
    clusteror = SpectralClustering(n_clusters=4, random_state=0)
    for user, photos in x_community.items():
        for photo, objects in photos.items():
            for object, scores in objects.items():
                if object in detectors:
                    for score in scores:
                        feature = [score, abs(detectors[object])]
                        features.append(feature)

    features = np.asarray(features)
    model = clusteror.fit(features)
    labels = model.labels_
    centers = model.cluster_centers_

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(features[:,0], features[:,1], c=labels, s=2)
    for i, j in centers:
        ax.scatter(i, j, s=50, c='red', marker='+')
    ax.set_xlabel('objectness')
    ax.set_ylabel('visual score')
    plt.colorbar(scatter)

    fig.savefig('vis_obj_'+situ_name+'.jpg')