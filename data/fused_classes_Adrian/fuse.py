import os
import copy

def fused_classes(file_1, file_2):
    """
    Combine classes from two files

    Parameters
    ----------
    file_1
    file_2

    Returns
    -------

    """
    class_file1 = []
    with open(file_1) as fp:
        lines = fp.readlines()

        for line in lines:
            parts = line.split('\t')
            class_name = parts[0]
            class_file1.append(class_name)

    class_file2 = []
    with open(file_2) as fp:
        lines = fp.readlines()

        for line in lines:
            parts = line.split('\n')
            class_name = parts[0]
            class_file2.append(class_name)

    fused_class = copy.deepcopy(class_file2)
    compare_purpose_class_file2 = []


    for syno_classes in class_file2:
            classes = syno_classes.split(',')

            for class_ in classes:
                compare_purpose_class_file2.append(class_)

    for class_ in class_file1:
        if class_ not in compare_purpose_class_file2:
            fused_class.append(class_)

    writer = open('fused_classes_v0.txt','w')
    for class_ in fused_class:
        writer.write('%s\n'%class_)

    writer.close()

if __name__ == '__main__':
    file_1 = 'label_statistics_ImgNet_OpenImage_v4.txt'
    file_2 = 'supplementary_classes_filtered.txt'

    fused_classes(file_1, file_2)
