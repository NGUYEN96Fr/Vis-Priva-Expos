import os
import shutil
import  json


def get_obj_list():
    """
    get list of new objects from .txt file

    Returns
    -------

    """
    txt_file = 'task_5_not_submitted_manual_label_stats.txt'
    obj_list = []
    syns = [] # synonym lists
    with open(txt_file) as fp:
        lines = fp.readlines()
        for line in lines:
            # in case having several synonyms indicating the same object, take the first
            class_names = line.split(',')
            if len(class_names) > 1:
                add_first = True
                for class_name in class_names:
                    if add_first:
                        not_spaced_name = class_name.replace(' ','_').split('\n')[0]
                        obj_list.append(not_spaced_name)
                        add_first = False
                    else:
                        not_spaced_name = class_name.replace(' ', '_').split('\n')[0]
                        syns.append(not_spaced_name)

            else:
                not_spaced_name = class_names[0].replace(' ','_').split('\n')[0]
                obj_list.append(not_spaced_name)

    return obj_list, syns


def retrieve_images(obj_list,syns):
    """

    Returns
    -------

    """
    limit = 300 # images
    cate_img_dict = {}
    save_dir = '/scratch_ssd/purpets_img_anns'
    taken_lists = [] # aldready considered objects

    def copy_images_1(path, cate, cate_img_dict):
        """copy images from directory containing images

            category path

        Returns
        -------

        """
        cate_path = os.path.join(path, cate)
        save_path = os.path.join(save_dir, cate)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if cate not in cate_img_dict:
            cate_img_dict[cate] = []

        images = os.listdir(cate_path)
        count = 0

        for image in images:
            count += 1
            if count <= limit:
                img_src = os.path.join(cate_path, image)
                img_dst = os.path.join(save_path, image)
                cate_img_dict[cate].append(img_src)
                shutil.copy(img_src, img_dst)
            else:
                break

        return cate_img_dict


    bing_path = '/scratch_global/vankhoa/bing_download/bing_images'
    flickr_path = '/scratch_global/vankhoa/flickr_download/flickr_images'
    paths = [flickr_path, bing_path]

    for path in paths:
        categories = os.listdir(path)
        for cate in categories:
            if cate in obj_list and cate not in taken_lists:
                cate_img_dict = copy_images_1(path, cate, cate_img_dict)
                taken_lists.append(cate)

    with open('retrieved_images.txt', 'w') as outfile:
        json.dump(cate_img_dict, outfile)


def main():
    obj_list, syns = get_obj_list()
    retrieve_images(obj_list, syns)


if __name__ == '__main__':
    main()