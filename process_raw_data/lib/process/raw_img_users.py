import os
import json

users_dict = {}
path = '/scratch_global/MAIF/exposure_prediction/gt_lists'

users = os.listdir(path)

for user in users:
    user_path = os.path.join(path,user)
    if user not in users_dict:
        users_dict[user] = []
    
    with open(user_path) as fp:
        lines = fp.readlines()

    for line in lines:
        parts = line.split('\t')
        img_id = parts[0]
        users_dict[user].append(img_id)

with open('raw_image_users.json', 'w') as fp:
    json.dump(users_dict,fp)
