"""
Create train and test sets of the mobinet inference, so the sets have
the same users in different partitions as the ones of the rcnn inference.

"""
import _init_paths
import os
import json
from process.user_profiles import retrieve_detected_objects

# CONFIGURATION
##########################################################################
########
# INPUTS
########
rcnn_file = '../out/train_test_split_rcnn_v1.json'
mobinet_inference = '../raw_data/inferences/v1/proc_gt_users_images_mobilenet.txt'

#########
# OUTPUTS
#########
out_file = '../out/train_test_split_mobinet_v1.json'
###########################################################################

train_test_rcnn = json.load(open(rcnn_file))
test_rcnn = train_test_rcnn['test']
train_rcnn = train_test_rcnn['train']
users = train_test_rcnn['users']
ratio_minibacthes = train_test_rcnn['ratio-minibacthes']

user_photos = retrieve_detected_objects(mobinet_inference)

mobinet_train_test_split = {}
mobinet_train_test_split['train'] = {}
mobinet_train_test_split['test'] = {}
mobinet_train_test_split['ratio_minibacthes'] = ratio_minibacthes
mobinet_train_test_split['users'] = []

for user in users:
    if user not in user_photos:
        raise ValueError('User does not exist')

mobinet_train_test_split['users'] = users

for percent, rcnn_user_photos in train_rcnn.items():
    mobinet_train_test_split['train'][percent] = {}
    for user, _ in rcnn_user_photos.items():
        mobinet_train_test_split['train'][percent][user] = user_photos[user]

for user, _ in test_rcnn.items():
    print()
    mobinet_train_test_split['test'][user] = user_photos[user]

print('Saving ...')
with open(out_file, 'w') as fp:
    json.dump(mobinet_train_test_split,fp)