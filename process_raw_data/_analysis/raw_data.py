import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def count_vis_parttern():
    """
    Count visual concept pattern
    :return:
    """
    path = '/home/nguyen/Documents/intern20/Vis-Priva-Expos/process_raw_data/raw_data/visual_concepts/processed_situations'

    vis_situs = os.listdir(path)
    vis_concepts = {}

    for vis_situ in vis_situs:
        print(vis_situ.split('.')[0])
        with open(os.path.join(path, vis_situ)) as fp:
            lines = fp.readlines()
            for line in lines:
                parts = line.split(' ')
                object_ = parts[0]
                score_ = float(parts[1].split('\n')[0])*3
                if object_ not in vis_concepts:
                    vis_concepts[object_] = []
                vis_concepts[object_].append(score_)

    # List number of patterns in visual concepts
    vis_patterns = {}
    stats = {}
    count = 0
    min_thresh = 0.38

    for concept_1, pattern_1 in vis_concepts.items():
        #print(concept_1)

        min_diff = 1000
        min_concepts = []

        for concept_2, pattern_2 in vis_concepts.items():
            residus = np.asarray(pattern_1) - np.asarray(pattern_2)
            diff = LA.norm(residus)

            if diff <= min_diff:
                if concept_2 != concept_1:
                    min_diff = diff

            if diff <= min_thresh:
                if concept_2 != concept_1:
                    min_concepts.append(concept_2)
        #print(min_diff)
        if min_diff < min_thresh:

            find_pattern = False
            find_p_ = ''
            for p_, concepts_ in stats.items():
                for min_concept in min_concepts:
                    if min_concept in concepts_:
                        find_pattern = True
                        find_p_ = p_

            if find_pattern:
                stats[find_p_].append(concept_1)
                vis_patterns[find_p_].append(pattern_1)

            else:
                count += 1
                vis_patterns['P' + str(count)] = []
                vis_patterns['P' + str(count)].append(pattern_1)
                stats['P' + str(count)] = []
                stats['P' + str(count)].append(concept_1)

        else:
            count += 1
            vis_patterns['P' + str(count)] = []
            vis_patterns['P' + str(count)].append(pattern_1)
            stats['P' + str(count)] = []
            stats['P' + str(count)].append(concept_1)

    #print(stats)
    #print(vis_patterns)

    vis_pattern_array = []

    for pattern, values in vis_patterns.items():
        mean_value = np.zeros(4)
        for value in values:
            mean_value = mean_value + np.asarray(value)

        mean_value = mean_value/len(values)
        vis_pattern_array.append(list(mean_value))

    vis_pattern_array = np.asarray(vis_pattern_array)
    vis_pattern_array = vis_pattern_array.transpose()
    print(stats)
    print('# patterns: ', len(vis_patterns))


    return vis_pattern_array, stats

def kmeans_pattern():
    """

    """
    from sklearn.cluster import KMeans

    # path = '/home/nguyen/Documents/intern20/Vis-Priva-Expos/process_raw_data/raw_data/visual_concepts/processed_situations'
    path = '/home/nguyen/Documents/intern20/Vis-Priva-Expos/process_raw_data/raw_data/user_exposures/v1/processed'
    K = 57 # number of clusters

    vis_situs = os.listdir(path)
    vis_concepts = {}
    situ_encode = {'job_search_IT.txt': 2,'bank_credit.txt': 1,'job_search_waiter_waitress.txt':3,'accommodation_search.txt':0}

    for vis_situ in vis_situs:
        print(vis_situ.split('.')[0])
        with open(os.path.join(path, vis_situ)) as fp:
            lines = fp.readlines()
            for line in lines:
                parts = line.split(' ')
                object_ = parts[0]
                score_ = float(parts[1].split('\n')[0])*3
                if object_ not in vis_concepts:
                    vis_concepts[object_] = [0,0,0,0]
                vis_concepts[object_][situ_encode[vis_situ]] = score_

    features =[]
    for concept, scores in vis_concepts.items():
        features.append(scores)

    features = np.asarray(features)

    kmeans = KMeans(n_clusters=K, random_state=0, max_iter=500).fit(features)
    centers = kmeans.cluster_centers_
    centers = centers.transpose()
    print('min: ', np.min(centers))
    print('max: ', np.max(centers))

    stats = {}

    for concept, scores in vis_concepts.items():
        scores = np.asarray(scores)
        scores = scores.reshape(1, scores.shape[0])
        pattern = kmeans.predict(np.asarray(scores))[0]
        pattern = 'P'+str(pattern)
        if pattern not in stats:
            stats[pattern] = []
        stats[pattern].append(concept)

    sum_centers = np.sum(centers, axis=0)
    # sorted_indexes = np.argsort(sum_centers)[::-1]
    sorted_indexes = np.argsort(sum_centers)

    org_stats = {}
    sorted_centers = centers[:, sorted_indexes]
    count_pattern = 1

    for index in range(K):
        old_key = 'P'+str(sorted_indexes[index])
        new_key = 'P'+str(count_pattern)
        org_stats[new_key] = stats[old_key]
        count_pattern += 1 

    for pattern, concepts in org_stats.items():
        message = pattern+'&'
        ADD = False
        for concept in concepts:
            if not ADD:
                message = message+' '+concept.replace('_',' ')
                ADD = True
            else:
                message = message+', '+concept.replace('_',' ')
                
        print(message)

    return sorted_centers, org_stats


def proc_user():
    """

    Returns
    -------

    """
    situ_encode = {0: 'job_search_IT', 1: 'bank_credit', 2: 'job_search_waiter_waitress', 3: 'accommodation_search'}
    path = '/home/nguyen/Documents/intern20/Vis-Priva-Expos/process_raw_data/raw_data/user_exposures/v1'
    evaluators = [eva for eva in os.listdir(path) if '.txt' in eva]
    situ_dict = {}
    for eva in evaluators:
        with open(os.path.join(path,eva)) as fp:
            lines = fp.readlines()
            for line in lines:
                parts = line.split(' ')
                if parts[1] != '4':
                    situ_ = situ_encode[int(parts[1])]
                    user_ = parts[0]
                    score_ = int(parts[2].split('\n')[0])
                    if situ_ not in situ_dict:
                        situ_dict[situ_] = {}
                    if user_ not in situ_dict[situ_]:
                        situ_dict[situ_][user_] = []
                    situ_dict[situ_][user_].append(score_)

    save_path = os.path.join(path, 'processed')
    if not os.path.exists(os.path.join(path, 'processed')):
        os.mkdir(save_path)

    for situ, users in situ_dict.items():
        sav_file = os.path.join(save_path, situ+'.txt')
        writer = open(sav_file, 'w')
        for user, scores in users.items():
            mean_ = np.mean(np.asarray(scores) - 4)/3
            text_ = user+' '+str(mean_)+'\n'
            writer.write(text_)
        writer.close()
    # print(situ_dict['job_search_IT']['53678425@N00'])

def plot():
    """"""
    # vis_pattern_array, stats = count_vis_parttern()
    vis_pattern_array, stats = kmeans_pattern()

    situs = ['ACC','BANK','IT','WAIT']
    situ_spots = np.arange(0,len(situs),1)
    pattern_spots = np.arange(0,vis_pattern_array.shape[1],1)

    patterns = [p_+'('+str(len(concepts))+')' for p_, concepts in stats.items()]

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.pyplot import figure
    #plt.tick_params(axis = "x", which = "both", bottom = True, top = False)

    figure(num=1, figsize=(14, 5), dpi=120, facecolor='w', edgecolor='k')

    cmap = LinearSegmentedColormap.from_list(
        name='test',
        colors=['red', 'white', 'green']
    )
    shifted_cmap = shiftedColorMap(cmap, midpoint=0.3448, name='shifted')
    im = plt.matshow(vis_pattern_array, fignum= 1, cmap=shifted_cmap)
    fig = plt.gcf()
    fig.colorbar(im, orientation="horizontal", pad=0.2, shrink = 0.5)
    plt.yticks(situ_spots, situs)
    ax = plt.gca()
    plt.xticks(pattern_spots,patterns,rotation ='vertical')
    plt.margins(0, 0)

    ax.xaxis.tick_bottom()
    plt.savefig('user_kmeans.png', bbox_inches = 'tight',pad_inches = 0.05)



def main():
    plot()
    # proc_user()

if __name__ == '__main__':
    main()