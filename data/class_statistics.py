# -*- coding: utf-8 -*-
"""Docstring for the class_statistic.py module
The module does class statistic in number of images on several datasets OpenImageNet, ImageNet, and Place2.


Note
----


Attributes
----------
"""

import os
import sys
import json
import csv
import configparser
import numpy as np


def sel_classes(path):
    """select interresting classes from an input file

    Parameters
    ----------
    path : string
        path to the input file

    Returns
    -------
    list
        a list of retrived classes

    Notes
    -----
    interresting classes come from the #todo column

    """
    classes = []
    skip = True
    line = 0

    with open(path) as fp:
        filereader = csv.reader(fp, delimiter=',')
        for row in filereader:
            if not skip:
                if 'x' in row[7]:
                    classes.append(row[3].split("\n")[0].split(" (")[0].lower())
            else:
                if line >= 2:
                    skip = False
                line += 1
    return classes


def count_OPID(path, classes, savpath):
    """count nb of imgs of each selected class in the Open ImageNet dataset

    Parameters
    ----------
    path : string
        path to the Open ImageNet dataset

    savpath : string
        saving path

    classes : list
        list of selected classes

    Returns
    -------


    """
    ##get trainable classes
    tnclasses = []
    with open(path + 'classes-trainable.txt') as fp:
        lines = fp.readlines()
        for line in lines:
            tnclasses.append(line.split("\n")[0])

    ##get trainable class names
    tnnames = []
    with open(path + 'class-descriptions.csv') as fp:
        filereader = csv.reader(fp, delimiter=',')
        for row in filereader:
            if row[0] in tnclasses:
                tnnames.append(row[1].split(" (")[0].lower())

    ##add the interresting classes if they are in the OPID trainable classes
    opinicls = []
    enopinicls = {}  # encoded opinicls
    for iclass in classes:

        for tnname in tnnames:
            if iclass == tnname:
                if iclass not in opinicls:
                    opinicls.append(iclass)

                if iclass not in enopinicls:
                    enopinicls[iclass] = []

                enopinicls[iclass].append(tnclasses[tnnames.index(tnname)])

    ##statistic on human-verified labels
    human_labeled = ['test-annotations-human-imagelabels.csv', 'validation-annotations-human-imagelabels.csv', \
                     'train-annotations-human-imagelabels.csv']
    human_img_dict = {}
    human_stats = {}

    ##statistic on machine-generated labels
    machine_generated = ['train-annotations-machine-imagelabels.csv', 'test-annotations-machine-imagelabels.csv', \
                         'validation-annotations-machine-imagelabels.csv']
    machine_img_dict = {}
    machine_stats = {}

    for file_ in human_labeled:
        with open(path + file_) as fp:
            filereader = csv.reader(fp, delimiter=',')
            ##we skip the first line, and count img number of each class. Count only imgs
            # having true positif verification.
            skip = True
            for row in filereader:
                if skip:
                    skip = False
                else:
                    if row[3] == '1':
                        name_tmp = None
                        for iname, keys in enopinicls.items():
                            if row[2] in keys:
                                name_tmp = iname
                        if name_tmp:
                            if name_tmp not in human_img_dict:
                                human_img_dict[name_tmp] = [[], []]
                                human_stats[name_tmp] = 0
                            if row[2] not in human_img_dict[name_tmp][0]:
                                human_img_dict[name_tmp][0].append(row[2])
                            human_img_dict[name_tmp][1].append(row[0])
                            human_stats[name_tmp] += 1

    for file_ in machine_generated:

        with open(path + file_) as fp:
            filereader = csv.reader(fp, delimiter=',')
            skip = True
            for row in filereader:
                if skip:
                    skip = False
                else:
                    name_tmp = None
                    for iname, keys in enopinicls.items():
                        if row[2] in keys:
                            name_tmp = iname
                    if name_tmp:
                        if name_tmp not in machine_img_dict:
                            machine_img_dict[name_tmp] = [[], []]
                            machine_stats[name_tmp] = 0
                        if row[2] not in machine_img_dict[name_tmp][0]:
                            machine_img_dict[name_tmp][0].append(row[2])
                        machine_img_dict[name_tmp][1].append(row[0])
                        machine_stats[name_tmp] += 1

    ##saving files
    opin_path = savpath + 'OID/'
    if not os.path.exists(opin_path):
        os.makedirs(opin_path)

    with open(opin_path + 'human_verified_img_paths.json', 'w') as fp:
        json.dump(human_img_dict, fp)

    with open(opin_path + 'human_verified_label_statistics.json', 'w') as fp:
        json.dump(human_stats, fp)

    with open(opin_path + 'machine_generated_img_paths.json', 'w') as fp:
        json.dump(machine_img_dict, fp)

    with open(opin_path + 'machine_generated_label_statistics.json', 'w') as fp:
        json.dump(machine_stats, fp)


def count_IMNET(path, classes, savpath):
    """count nb of imgs on selected classes in the ImageNet dataset

    Parameters
    ----------
    path : string
        path to the Open ImageNet dataset

    savpath : string
        saving path

    classes : list
        list of selected classes


    Returns
    -------

    """
    ##get class name and class id
    name_id = {}

    with open(path + 'synsets_words_size_map.txt') as fp:
        lines = fp.readlines()

        for line in lines:
            id_ = line.split('\t')[0]
            name = line.split('\t')[1].split(' ')[0].replace('_', ' ').lower()
            img_nb = int(line.split('\t')[2])
            if name not in name_id:
                name_id[name] = [[id_], img_nb]
            else:
                name_id[name][0].append(id_)
                name_id[name][1] += img_nb

    inet_stat = {}

    for iclass in classes:
        if iclass in name_id:
            inet_stat[iclass] = name_id[iclass][1]

    inet_path = savpath + 'IMNET/'
    if not os.path.exists(inet_path):
        os.makedirs(inet_path)

    with open(inet_path + 'imnet_statistics.json', 'w') as fp:
        json.dump(inet_stat, fp)


def to_txt(savpath, classes):
    """class label statistic summary

    Parameters
    ----------
    savpath : string
        class label dataset stats

    classes : list
        list of selected classes

    Returns
    -------

    """
    with open(savpath + 'IMNET' + '/imnet_statistics.json') as fp0:
        imnet = json.loads(fp0.read())
    with open(savpath + 'OID' + '/human_verified_label_statistics.json') as fp1:
        oid1 = json.loads(fp1.read())
    with open(savpath + 'OID' + '/machine_generated_label_statistics.json') as fp2:
        oid2 = json.loads(fp2.read())

    stat = {}
    ranked_label = []
    sum_ = []

    for iclass in classes:
        nb_img = []

        if iclass in imnet:
            nb_img.append(imnet[iclass])
        else:
            nb_img.append(0)

        if iclass in oid1:
            nb_img.append(oid1[iclass])
        else:
            nb_img.append(0)

        if iclass in oid2:
            nb_img.append(oid2[iclass])
        else:
            nb_img.append(0)

        sum_.append(sum(nb_img))
        ranked_label.append(iclass)
        nb_img.append(sum(nb_img))
        stat[iclass] = (nb_img[0], nb_img[1], nb_img[2], nb_img[3])

    sorted_indexes = list(np.argsort(sum_)[::-1])

    writer = open('label_statistics_ImgNet_OpenImage_v4.txt', 'w')
    writer.write('label' + '\t' + 'IMNET' + '\t' + 'OID1' + '\t' + 'OID2' + '\t' + 'Sum\n')

    for index in sorted_indexes:
        label_ = ranked_label[index]
        img_nb = stat[label_]
        writer.write('%s\t%s\t%s\t%s\t%s\n' % (label_, img_nb[0], img_nb[1], img_nb[2], img_nb[3]))

    writer.close()


def main():
    """


    """
    ##read the parameter config file
    conf = configparser.ConfigParser()
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]

    ##get params
    cpath = conf['cpath']
    opidpath = conf['opidpath']
    inetpath = conf['inetpath']
    savpath = conf['savpath']

    ##get a list of interresting classes
    classes = sel_classes(cpath)

    ##search the classes in the Open ImageNet dataset
    count_OPID(opidpath, classes, savpath)

    ##search the classes in the ImageNet dataset
    count_IMNET(inetpath, classes, savpath)

    ##to a .txt file, statistic on the datasets
    to_txt(savpath, classes)


if __name__ == "__main__":
    main()