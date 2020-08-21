"""
The file retrieves images from ImageNet and Open Image
dataset from data in Bergamote server.

"""
import os
import sys
import shutil
import configparser
import json
import tqdm
import numpy as np
import tarfile


def read_label(path, dataset):
    """Read retrieved labels from the passed file

    :param path : string
        a path to the label statistics file
    :param dataset : string
        dataset name (IMNET, OID1, OID2)
        + OID1 : human verified labels
        + OID2 : machine generated labels

    :returns
        labels_ds : dict
            labels in the dataset and its number of images
    """
    labels_ds = {}
    skip = True

    with open(path) as fp:
        lines = fp.readlines()
        for line in lines:
            if skip:
                skip = False
            else:
                parts = line.split('\t')
                if dataset == 'IMNET':
                    if int(parts[1]) > 0:
                        labels_ds[parts[0]] = int(parts[1])
                elif dataset == 'OID1':
                    if int(parts[2]) > 0:
                        labels_ds[parts[0]] = int(parts[2])
                elif dataset == 'OID2':
                    if int(parts[3]) > 0:
                        labels_ds[parts[0]] = int(parts[3])
                else:
                    raise ValueError('Wrong dataset keyword')

    return labels_ds


def label_key_mapping(labels_ds, path, dataset):
    """
    :param labels_ds: string
        labels in the dataset

    :param path: string
        path to the key file

    :param dataset : string
        dataset name

    :returns
        keys_ds : dict
            key classes and its nb of images
    """
    keys_ds = {}
    if dataset == 'IMNET':
        with open(path) as fp:
            classes = fp.readlines()
            for class_ in classes:
                name = class_.split('\t')[1].split(' ')[0].replace('_', ' ').lower()
                key = class_.split('\t')[0]
                nb = int(class_.split('\t')[2])
                if name in labels_ds:
                    if name not in keys_ds:
                        keys_ds[name] = [[key], nb]
                    else:
                        keys_ds[name][0].append(key)
                        keys_ds[name][1] += nb

        assert len(labels_ds) == len(keys_ds), \
            'len(labels): %d != len(keys): %d' % (len(labels_ds), len(keys_ds))

        for name, nb1 in labels_ds.items():
            nb2 = keys_ds[name][1]
            if nb1 != nb2:
                raise RuntimeError('nb of images not equal ! ', nb1, '!=', nb2)

    if dataset == 'OID1' or dataset == 'OID2':

        with open(path) as fp:
            keys_ds = json.loads(fp.read())

    return keys_ds


def retrieve_img(dataset, keys, img_path, sel_classes,heavy_file_path):
    """retrieve images by keys

    :param img_path: string
        an image location path

    :param dataset : string
        dataset name

    :param keys : dict

    :param sel_classes : list
        list of selected classes

    :return: dict
        retrieved images saved in many individual folders named by class names

    """

    image_paths = {}

    if dataset == 'IMNET':
        # get list of keys saved on the bergamote cluster.
        keys_bergamote = [key.split('.')[0] for key in os.listdir(img_path)]
        for label, keys_nb in keys.items():
            if label in sel_classes:
                if label not in image_paths:
                    image_paths[label] = []
                keys = keys_nb[0]
                for key in keys:
                    if key not in keys_bergamote:
                        raise ValueError(key, 'not exist in Bergamote')
                    with open(os.path.join(img_path, key + '.lst')) as fp:
                        lines = fp.readlines()
                        for line in lines:
                            image_paths[label].append(line.replace('\n',''))

    elif dataset == 'OID1' or dataset == 'OID2':
        for label, keys_imgs in keys.items():
            if label in sel_classes:
                nb_img_not_exist = 0
                if label not in image_paths:
                    image_paths[label] = []
                imgs = keys_imgs[1]
                print(label)
                for img in tqdm.tqdm(imgs):
                    # check if the image exists
                    sub_folder = img[:4]
                    if os.path.exists(os.path.join(img_path, sub_folder, img + '.jpg')):
                        image_paths[label].append(os.path.join(img_path, sub_folder, img + '.jpg'))
                    else:
                        #raise RuntimeError(img + '.jpg does not exist!')
                        nb_img_not_exist += 1
                print('not exist: ',nb_img_not_exist,'/',len(imgs))


    with open(heavy_file_path + '/image_paths_'+str(dataset)+'.json', 'w') as fp0:
        json.dump(image_paths, fp0)

    if dataset == 'IMNET':
        print(image_paths)

    return image_paths


def combine_images(paths, save_path, heavy_file_path, nb_abb, class_maxsize, sel_classes, voi, manual_retrieve, auto_retrieve):
    """Combine images from IMNET, OID1, OID2 to form two distinct sets
    ,including a manual annotation set, and a automatic annotation set.

    :param save_path : string

    :param nb_abb : int
        nb of images to manually annotate bounding boxes
        per each class

    :param class_maxsize : int
        max number of images per a class in automatic annotation

    :param  paths : list
        paths = [imnet_paths, oid1_paths, oid2_paths]

    :param sel_classes : list
        list of selected classes

    :param voi : string
        version of the openimage

    :param auto_batch : int
        batch size of automatic annotation for each category

    :param manual_retrieve : boolean
        if retrieve manual images

    :param auto_retrieve : boolean
        if retrieve auto images

    :return
        auto : dict
         images used for automatic annotations

        manual : dict
         images used for manual annotations

    """
    auto = {}
    manual = {}

    for img_path_dict in paths:
        for label, img_paths in img_path_dict.items():
            if label not in auto:
                auto[label] = []
                manual[label] = []

            # priority in selecting images
            # in oid1, imnet, oid2 for manually annotate
            for img_path in img_paths:
                if len(manual[label]) < nb_abb:
                    manual[label].append(img_path)
                else:
                    auto[label].append(img_path)

    with open(heavy_file_path + '/auto_anno_imgPaths_Imnet_openImg_'+voi+'.json', 'w') as fp0:
        json.dump(auto, fp0)

    with open(heavy_file_path+'/manual_anno_imgPaths_Imnet_openImg_'+voi+'.json', 'w') as fp1:
        json.dump(manual, fp1)

    ################################
    writer = open('selected_classes_Statistics_Imnet_openImg_'+voi+'.txt','w')
    writer.write('class\tmanual\tauto\tsum\n')
    nb_auto = []
    nb_manual = []
    for class_ in sel_classes:

        if class_ in auto:
            nb_auto.append(len(auto[class_]))
        else:
            nb_auto.append(0)

        if class_ in manual:
            nb_manual.append(len(manual[class_]))
        else:
            nb_manual.append(0)

    nb_sum = [nb_auto[i]+nb_manual[i] for i in range(len(sel_classes))]
    sorted_indexes = list(np.argsort(nb_sum)[::-1])

    for index in sorted_indexes:
        writer.write('%s\t%s\t%s\t%s\n'%(sel_classes[index],nb_manual[index],nb_auto[index],nb_sum[index]))
    writer.close()

    #################################
    print('Save to directories !')
    extract_tar_IMNET = [] # untared prefix files of ImageNet
    auto_list = []
    auto_dict = {}
    manual_list = []

    if save_path:

        auto_path = os.path.join(save_path, 'auto')
        manual_path = os.path.join(save_path, 'manual')
        if not os.path.exists(save_path+'/'+'auto') and not os.path.exists(save_path+'/'+'manual'):
            os.mkdir(auto_path)
            os.mkdir(manual_path)

        print('Manual annotations !')
        if manual_retrieve:
            for class_, img_srcs in manual.items():
                class_path = os.path.join(manual_path, class_.replace(' ', '_'))
                if not os.path.exists(class_path):
                    os.mkdir(class_path)
                for img_src in tqdm.tqdm(img_srcs):
                    ## Extract .tar files of ImgNet to a scratch_global/vankhoa/..
                    if 'ImageNet' in img_src:
                        img_tar = img_src.split('/')[-2] +'.tar'
                        if img_tar not in extract_tar_IMNET:
                            extract_tar_IMNET.append(img_tar)
                            df = tarfile.open('/scratch_global/DATASETS/ImageNet/tars/'+img_tar)
                            df.extractall(path = '/scratch_global/vankhoa/IMAGENET')
                        new_img_src = '/scratch_global/vankhoa/IMAGENET' + '/'+img_src.split('/')[-1]

                    else:
                        new_img_src = img_src
                    manual_list.append(new_img_src)
                    shutil.copy(new_img_src, class_path +'/'+ img_src.split('/')[-1])

        print('Automatic annotations !')
        if auto_retrieve:
            txt_main_path = '/scratch_global/vankhoa/official_train_inference_ann_models/ann_infer'
            img_infer_train_save_dir = '/scratch_global/vankhoa/official_train_inference_ann_models/images'

            for class_, img_srcs in auto.items():

                print(class_)
                auto_dict[class_] = []
                class_path = os.path.join(auto_path, class_.replace(' ', '_'))
                class_img_count = 0
                if not os.path.exists(class_path):
                    os.mkdir(class_path)

                ##############
                class_txt_file = os.path.join(txt_main_path, class_.replace(' ', '_') + '.txt')
                class_writer = open(class_txt_file, 'w')
                ##############

                for img_src in tqdm.tqdm(img_srcs):
                    class_img_count += 1
                    if class_img_count <= class_maxsize:
                        ## Extract .tar files of ImgNet to a scratch_global/vankhoa/..
                        if 'ImageNet' in img_src:
                            img_tar = img_src.split('/')[-2] +'.tar'
                            if img_tar not in extract_tar_IMNET:
                                extract_tar_IMNET.append(img_tar)
                                df = tarfile.open('/scratch_global/DATASETS/ImageNet/tars/'+img_tar)
                                df.extractall(path = '/scratch_global/vankhoa/IMAGENET')
                            new_img_src = '/scratch_global/vankhoa/IMAGENET' +'/'+ img_src.split('/')[-1]
                        else:
                            new_img_src = img_src

                        auto_list.append(new_img_src)
                        # shutil.copy(new_img_src, class_path +'/'+ img_src.split('/')[-1])

                        #########
                        new_img_dst = os.path.join(img_infer_train_save_dir, img_src.split('/')[-1])
                        auto_dict[class_].append(new_img_dst)
                        if not os.path.exists(new_img_dst):
                            shutil.copy(new_img_src, new_img_dst)
                        class_writer.write('%s\n'%new_img_dst)
                        #########

                    else:
                        break

                class_writer.close()

        ## generate retrieved image lists
        with open(heavy_file_path + '/retrieved_manual_img_paths_ImgNet_OpenImage_' + voi + '.json', 'w') as fp1:
            json.dump(manual_list, fp1)

        with open(heavy_file_path + '/retrieved_auto_img_paths_ImgNet_OpenImage_' + voi + '.json', 'w') as fp0:
            json.dump(auto_list, fp0)

        #############
        writer_supp = open('../fused_classes_Adrian/classifying_tasks/30_infer_class_stats.txt', 'w')
        for class_, imgs in auto_dict.items():
            writer_supp.write('%s\\%s\n'%(class_,len(imgs)))
        writer_supp.close()
        #############

def main():
    # read the param config file

    conf = configparser.ConfigParser()
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]

    # get params
    label_path = conf['label_path']
    nb_abb = int(conf['nb_abb'])
    sel_path = conf['sel_class_path']
    save_path = conf['save_path']
    load = conf.getboolean('load') #load image paths
    voi = conf['voi'] #version of the openimage
    manual_retrieve = conf.getboolean('manual_retrieve')
    auto_retrieve = conf.getboolean('auto_retrieve')
    class_maxsize = conf.getint('class_maxsize')
    heavy_file_path = conf['heavy_file_path']

    if not os.path.exists(heavy_file_path):
        os.mkdir(heavy_file_path)

    # create the saved directory
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # read selected classes
    sel_classes = []
    with open(sel_path) as fp:
        classes = fp.readlines()
        for class_ in classes:
            sel_classes.append(class_.replace('\n',''))

    # get labels and its corresponding image paths
    if not load:
        imnet_labels = read_label(label_path, 'IMNET')
        imnet_keys = label_key_mapping(imnet_labels, '/home/vankhoa/DATA/IMNET_Sources/synsets_words_size_map.txt', 'IMNET')
        imnet_path = retrieve_img('IMNET', imnet_keys, '/scratch_global/DATASETS/ImageNet/imageLists',sel_classes,heavy_file_path)

        oid1_labels = read_label(label_path, 'OID1')
        oid1_keys = label_key_mapping(oid1_labels,heavy_file_path + '/Statistics_ImgNet_OpenImage_'+voi+'/OID/human_verified_img_paths.json', 'OID1')
        oid1_path = retrieve_img('OID1', oid1_keys, '/scratch_global/vankhoa/images',sel_classes,heavy_file_path)

        oid2_labels = read_label(label_path, 'OID2')
        oid2_keys = label_key_mapping(oid2_labels,heavy_file_path + '/Statistics_ImgNet_OpenImage_'+voi+'/OID/machine_generated_img_paths.json', 'OID2')
        oid2_path = retrieve_img('OID2',oid2_keys, '/scratch_global/vankhoa/images',sel_classes,heavy_file_path)

    else:
        with open(heavy_file_path + '/image_paths_' +'IMNET' + '.json', 'r') as fp0:
            imnet_path = json.loads(fp0.read())

        with open(heavy_file_path + '/image_paths_' +'OID1' + '.json', 'r') as fp1:
            oid1_path = json.loads(fp1.read())

        with open(heavy_file_path + '/image_paths_' +'OID2' + '.json', 'r') as fp2:
            oid2_path = json.loads(fp2.read())

    # combine images from ImageNet and Open Image and retrieve them.
    combine_images([oid1_path, imnet_path, oid2_path], save_path, heavy_file_path, nb_abb, class_maxsize, sel_classes, voi, manual_retrieve, auto_retrieve)


if __name__ == '__main__':
    main()