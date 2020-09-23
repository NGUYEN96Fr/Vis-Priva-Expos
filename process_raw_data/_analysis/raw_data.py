import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def count_vis_parttern():
    """
    Count visual concept pattern
    :return:
    """
    path = '/home/users/vnguyen/intern20/Vis-Priva-Expos/' \
           'process_raw_data/raw_data/visual_concepts/processed_situations'

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
    min_thresh = 0.4

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
                vis_patterns['p_' + str(count)] = []
                vis_patterns['p_' + str(count)].append(pattern_1)
                stats['p_' + str(count)] = []
                stats['p_' + str(count)].append(concept_1)

        else:
            count += 1
            vis_patterns['p_' + str(count)] = []
            vis_patterns['p_' + str(count)].append(pattern_1)
            stats['p_' + str(count)] = []
            stats['p_' + str(count)].append(concept_1)

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


def plot_():
    """"""
    vis_pattern_array, stats = count_vis_parttern()

    situs = ['IT', 'BANK', 'ACCOM', 'WAIT']
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
    plt.matshow(vis_pattern_array, fignum= 1, cmap=cmap)
    plt.yticks(situ_spots, situs)
    ax = plt.gca()
    plt.xticks(pattern_spots,patterns,rotation ='vertical')

    ax.xaxis.tick_bottom()
    plt.savefig('abc.jpg')


def main():
    plot_()

if __name__ == '__main__':
    main()