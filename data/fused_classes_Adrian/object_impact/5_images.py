"""
This module prepares 5 images for object impact scores per situation.

"""
import  os
import shutil

def get_obj_list():
    """
    get list of new objects from .txt file

    Returns
    -------

    """
    txt_file = 'fused_class_bboxable_stats.txt'
    obj_list = []
    syns = [] # synonym lists
    with open(txt_file) as fp:
        lines = fp.readlines()
        for line in lines:
            # in case having several synonyms indicating the same object, take the first
            class_names = line.split('\\')[0].split(',')
            if len(class_names) > 1:
                add_first = True
                for class_name in class_names:
                    if add_first:
                        not_spaced_name = class_name.replace(' ','_')
                        obj_list.append(not_spaced_name)
                        add_first = False
                    else:
                        not_spaced_name = class_name.replace(' ', '_')
                        syns.append(not_spaced_name)

            else:
                not_spaced_name = class_names[0].replace(' ','_')
                obj_list.append(not_spaced_name)

    return obj_list, syns


def retrieve_images(obj_list,syns):
    """

    Parameters
    ----------
    obj_list

    Returns
    -------

    """
    limit = 10 # images
    save_dir = '/scratch_ssd/purpets_object_impact'
    taken_lists = [] # aldready considered objects
    print('COPY 1')
    def copy_images_1(path, cate):
        """copy images from directory containing images

            category path

        Returns
        -------

        """
        cate_path = os.path.join(path, cate)
        save_path = os.path.join(save_dir, cate)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        images = os.listdir(cate_path)
        count = 0

        for image in images:
            count += 1
            if count <= limit:
                img_src = os.path.join(cate_path, image)
                img_dst = os.path.join(save_path, image)
                shutil.copy(img_src, img_dst)
            else:
                break

    bing_path = '/scratch_global/vankhoa/bing_download/bing_images'
    flickr_path = '/scratch_global/vankhoa/flickr_download/flickr_images'
    paths = [flickr_path, bing_path]


    for path in paths:
        categories = os.listdir(path)
        for cate in categories:
            if cate in obj_list and cate not in taken_lists:
                copy_images_1(path, cate)
                taken_lists.append(cate)

    print('COPY 2')
    def copy_image_2(path, cat_txt):
        """

        Returns
        -------

        """
        cate = cat_txt.split('.')[0].replace(' ', '_')
        cat_txt_path = os.path.join(path, cat_txt)
        save_path = os.path.join(path, cate)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        count = 0

        with open(cat_txt_path) as fp:
            lines = fp.readlines()
            for line in lines:
                count += 1
                if count <= limit:
                    img_name = line.split('/')[-1].split('\n')[0]
                    src_image = line.split('\n')[0]
                    dst_image = os.path.join(save_path,img_name)
                    shutil.copy(src_image, dst_image)
                else:
                    break


    ann_infer = '/scratch_global/vankhoa/official_train_inference_ann_models/ann_infer'
    ann_training = '/scratch_global/vankhoa/official_train_inference_ann_models/ann_training'
    dect_training = '/scratch_global/vankhoa/official_train_inference_ann_models/dect_training'
    paths = [ann_infer, ann_training, dect_training]

    for path in paths:
        categories = os.listdir(path)
        for cat_txt in categories:
            cate = cat_txt.split('.')[0].replace(' ','_')
            if cate in obj_list and cate not in taken_lists:
                copy_image_2(path, cat_txt)
                taken_lists.append(cate)

def main():
    """

    Returns
    -------

    """
    obj_list, syns = get_obj_list()
    print(syns)
    retrieve_images(obj_list,syns)


if __name__ == '__main__':
    main()