import os

"""
This module manages classes into different categories:

    1. Classes whose images have bounding boxes (coco, imnet, openimg) (> 250 images & 2000 images).
        file_name: task_1_suff_boxable_classes.txt (sufficient boxable classes)

    2. Classes whose images have bounding boxes (coco, imnet, openimg) (< 2000 images & > 0 images).
        
        file_name: task_2_not_suff_boxable_classes.txt
        
        To train detection models, we take at least 2000 images per classes
        2.1 classes have sufficient images to train detection models (> 2000 images)
            task_2-1_suff_boxable_detection.txt

    3. Classes were already submitted to annotators (> 200 images & < 300 images)
        file_name: submitted_manual_label_statistics.txt

    4. Classes have enough annotated images to train annotating models (> 200 images < 500 images)
        file_name: suff_image_ann_training.txt

    5. Classes are not submitted to annotators yet.
        file_name: 
            not_submitted_manual_label_stats.txt

    6. Classes have enough images for the automatic annotation task (or inference) >= 3000 images
        file_name: suff_image_inference.txt

    7. Classes need to be searched on other type datasets (classification, ...) or Bing, or Flickr
        file_name: class_searching_bing.txt
                   class_searching_flickr.txt
                   class_searching_datasets.txt
    

"""

def read_file():
    """

    Returns
    -------
    - boxable_cls_stats = {}
    - submitted_classes = {}
    """

    boxable_cls_stats = {}
    submitted_classes = {}

    with open('fused_class_bboxable_stats.txt') as fp:
        lines = fp.readlines()
        ignore = True
        for line in lines:
            if ignore:
                ignore = False
            else:
                parts = line.split('\\')
                sum_ = int(parts[-1])
                class_ = parts[0]
                boxable_cls_stats[class_] = sum_

    with open('submitted_manual_label_statistics.txt') as fp1:
        lines = fp1.readlines()
        for line in lines:
            parts = line.split('-')
            nb_img = int(parts[-1])
            class_ = parts[0].replace('_', ' ')
            submitted_classes[class_] = nb_img

    return boxable_cls_stats, submitted_classes


def task_1_2(boxable_cls_stats, K1 = 250, K2 = 2000):
    """
    1. Classes whose images have bounding boxes (coco, imnet, openimg) (> 250 images & 2000 images).
    2. Classes whose images have bounding boxes (coco, imnet, openimg) (< 2000 images & > 0 images).

    Parameters
    ----------
    boxable_cls_stats

    Returns
    -------
        task1_classes = {}
        task2_classes = {}

    """
    task1_classes = {}
    task2_classes = {}
    task2_1_classes = {}

    writer1 = open('task_1_suff_boxable_classes.txt','w')
    writer1.write('class\\nb_imgs\n')
    writer2 = open('task_2_not_suff_boxable_classes.txt','w')
    writer2.write('class\\existing_imgs\n')
    writer2_1 = open('task_2-1_suff_boxable_detection.txt','w')
    writer2_1.write('class\\existing_imgs\n')
    for cls, nb_img in boxable_cls_stats.items():

        if nb_img > K1 and nb_img < K2:
            writer1.write('%s\\%s\n'%(cls, nb_img))
            task1_classes[cls] = nb_img

        if nb_img < K2: ## need to add more manually annotated
                        # images by running inferences on trained annotating models
            writer2.write('%s\\%s\n'%(cls, nb_img))
            task2_classes[cls] = 0

        if nb_img >= K2:
            writer2_1.write('%s\\%s\n'%(cls, nb_img))
            task2_1_classes[cls] = nb_img

    writer1.close()
    writer2.close()

    return task1_classes, task2_classes, task2_1_classes

def task_3():
    """

    Returns
    -------

    """
    pass

def task_4(submitted_classes,task1_classes):
    """
    Classes have enough annotated images to train annotating models

    Parameters
    ----------
    submitted_classes
    task1_classes

    Returns
    -------
        suff_img_training: dict

    """
    suff_img_training = {}
    for class_, nb_img in submitted_classes.items():
        if class_ not in suff_img_training:
            suff_img_training[class_] = 0
        suff_img_training[class_] += nb_img

    for class_, nb_img in task1_classes.items():
        if class_ not in suff_img_training:
            suff_img_training[class_] = 0
        suff_img_training[class_] += nb_img

    writer = open('task_4_suff_img_ann_training.txt','w')
    writer.write('class\\nb_imgs\n')

    for class_, nb_img in suff_img_training.items():
        writer.write('%s\\%s\n'%(class_,nb_img))

    writer.close()

    return suff_img_training


def task_5_6_7(suff_img_training, task2_classes, K = 500):
    """
    5. Classes have enough images but not yet submitted to annotators
        file_name: not_submitted_manual_label_stats.txt

    6. Classes have enough images for the automatic annotation task (or inference) >= 3000 images + 250 training
        file_name: task_6_suff_image_training_inference.txt

    7. Classes need to be searched on other type datasets (classification, ...) or Bing, or Flickr
        file_name: task_7_class_searching_bing.txt
                   task_7_class_searching_flickr.txt
                   task_7_class_searching_datasets.txt
    Returns
    -------

    """
    bing_classes_pc_path = '/home/nguyen/Documents/intern20/NO_DELETE_bing_img_download/data/bing_images'
    classes_PC = os.listdir(bing_classes_pc_path) # already existing classes on PC downloaded from Bing
    existing_bing_classes_PC = {}
    for class_ in classes_PC:
        nb_img = len(os.listdir(os.path.join(bing_classes_pc_path,class_)))
        class_name = class_.replace('_',' ')
        existing_bing_classes_PC[class_name] = nb_img

    existing_classes_datasets = {} # already existing classes on classification datasets on the Bergamote server
    with open('label_statistics_ImgNet_OpenImage_v4.txt') as fp:
        lines = fp.readlines()
        ignore = True
        for line in lines:
            if ignore:
                ignore = False
            else:
                parts = line.split('\t')
                class_name = parts[0]
                nb_img = int(parts[-1])
                existing_classes_datasets[class_name] = nb_img

    for class_, nb_img in task2_classes.items():
        if class_ in existing_bing_classes_PC:
            task2_classes[class_] += existing_bing_classes_PC[class_]

        if class_ in existing_classes_datasets:
            task2_classes[class_] += existing_classes_datasets[class_]

    task5_writer = open('task_5_not_submitted_manual_label_stats.txt','w')
    task6_writer = open('task_6_suff_image_training_inference.txt','w')
    task7_bing = open('task_7_class_searching_bing.txt','w')
    task7_flickr = open('task_7_class_searching_flickr.txt','w')
    task7_datasets = open('task_7_class_searching_datasets.txt','w')
    for class_, nb_img in task2_classes.items():

        if class_ not in suff_img_training:
            task5_writer.write('%s\n' % class_)

        if nb_img >= K:
            task6_writer.write('%s\\%s\n'%(class_,nb_img))

        else:
            task7_flickr.write('%s\n'%class_)

            if class_ not in existing_bing_classes_PC:
                task7_bing.write('%s\n'%class_)

            if class_ not in existing_classes_datasets:
                task7_datasets.write('%s\n'%class_)

    task5_writer.close()
    task6_writer.close()
    task7_datasets.close()

def main():
    """

    Returns
    -------

    """
    boxable_cls_stats, submitted_classes = read_file()
    task1_classes, task2_classes, task2_1_classes = task_1_2(boxable_cls_stats, K1=250, K2=2000)
    suff_img_training = task_4(submitted_classes, task1_classes)
    task_5_6_7(suff_img_training, task2_classes, K=500)

if __name__ == '__main__':
    main()


