"""
This modules retrives necessary objects for manual annotation

"""

import os
import json

path = 'output_class_priv'
path1 = 'old_objects'

difficult_objects = []

def object_rating():
    """
    Read object impact rating

    0: IT, 1: Bank, 2:Waiter, 3:Acc, 4:life insurance
    Returns
    -------

    """
    evaluators = os.listdir(path)

    Stats = {}
    for eval in evaluators:
        if '0' in eval or '1' in eval or '3' in eval:
            with open(os.path.join(path,eval)) as jfile:
                data = json.load(jfile)

            k = eval.split('.')[0].split('_')[-1]
            for cate, score in data[k]['labels'].items():
                if cate not in Stats:
                    Stats[cate] = 0

                Stats[cate] += abs(score - 4)

    Stats = {k: v for k, v in sorted(Stats.items(), key=lambda item: item[1])[::-1]}

    return Stats



def old_object():
    """
    Read already annotated objects

    Returns
    -------

    """
    old_objects = []
    files = os.listdir(path1)
    for file in files:
        with open(os.path.join(path1,file)) as fp:
            lines = fp.readlines()

            for line in lines:
                objects = line.split('\n')[0].split('\\')[0].split(',')
                for object_ in objects:
                    old_objects.append(object_.replace(' ','_'))

    return old_objects


def retrieve_prior_objects(sorted_rating, old_objects, limit, save_file):
    """

    Parameters
    ----------
    sorted_rating
    old_objects

    Returns
    -------

    """
    prior_objects = {}
    count = 0
    writer = open(os.path.join(path1,save_file),'w')

    for object, score in sorted_rating.items():
        if object not in difficult_objects:
            if object not in old_objects and count <= limit:
                prior_objects[object] = score
                count += 1
                writer.write('%s\n'%object)

    writer.close()
    return prior_objects

def set_outfile():
    """

    Returns
    -------

    """
    files = os.listdir(path1)
    max_current_version = -1
    for file in files:
        if 'prior_object' in file:
            version = int(file.split('.')[0].split('_')[-1].split('v')[-1])
            if version > max_current_version:
                max_current_version = version
    out_file = 'prior_object_v'+str(max_current_version+1)

    return out_file



def main():
    """

    Returns
    -------

    """
    limit = 20  # 10 prior objects

    save_file = set_outfile()
    sorted_rating = object_rating()
    old_objects = old_object()
    prior_objects = retrieve_prior_objects(sorted_rating, old_objects, limit, save_file)


if __name__ == '__main__':
    main()