"""
Pre-process raw inference files

"""
import os
import json
import pickle


path = '/home/users/vnguyen/intern20/Vis-Priva-Expos/process_raw_data/raw_data/inferences/v1'
init_infer_path = 'gt_users_images_mobilenet.txt'
proc_infer_path = 'proc_gt_users_images_mobilenet.txt'

image_user_path = 'raw_image_users.pkl'
#image_user_path = 'raw_image_users.json'
img_not_found = []

with open(os.path.join(path,image_user_path), 'rb') as pkl_file:
    img_users = pickle.load(pkl_file)

# with open(os.path.join(path,image_user_path), 'rb') as json_file:
#     img_users = json.load(json_file)

writer = open(os.path.join(path, proc_infer_path), 'w')

with open(os.path.join(path,init_infer_path)) as fp:
    lines = fp.readlines()
    for line in lines:
        im_id = line.split(' ')[0]
        user_id = ''
        find_user = False
        for user, img_ids in img_users.items():
            if im_id in img_ids:
                user_id = user
                find_user = True
                break
        if not find_user:
            if im_id not in img_not_found:
                img_not_found.append(im_id)
        else:
            writer.write('%s %s'%(user_id,line))

writer.close()
print('NB of images not found: ', len(img_not_found))

with open('img_not_found.json', 'w') as fp:
    json.dump(img_not_found,fp)




