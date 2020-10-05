"""
Create ground-truth user exposures
for training the VISPEL model

"""

import _init_paths
import os
import json
from process.user_expos import load_gt_user_profiles, normalizing

# CONFIGURATION
##########################################################################
########
# INPUTS
########
normalize = True
rcnn_file = '../out/train_test_split_rcnn_v1.json'
user_expo_paths = '../raw_data/user_exposures/v1'
#########
# OUTPUTS
#########
expo_out_file = '../out/gt_usr_exposure_v1.json'
############################################################################

train_test_rcnn = json.load(open(rcnn_file))
valid_users = train_test_rcnn['users']

if normalize:
    norm_expo_paths = os.path.join(user_expo_paths, 'normalized_expos')
    if not os.path.exists(norm_expo_paths):
        os.makedirs(norm_expo_paths)
    normalizing(user_expo_paths, norm_expo_paths)
    user_score_situs = load_gt_user_profiles(norm_expo_paths, valid_users, normalize)

else:
    user_score_situs = load_gt_user_profiles(user_expo_paths, valid_users, normalize)

print('Situations: ')
for situ, scores in user_score_situs.items():
    print('\t', situ.replace('_', ' '), ' nb_users = ', len(list(scores.keys())))

print('Saving ...')
with open(expo_out_file, 'w') as fp1:
    json.dump(user_score_situs, fp1)
print('Done!')
