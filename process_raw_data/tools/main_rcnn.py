"""
The module create train and test sets by basing on the
rcnn inferences.

Usage:
    python main_rcnn.py
"""

import _init_paths
import json
from data.train_test_split import train_test_split
from process.user_profiles import retrieve_detected_objects

def main():
    """
    
    """
    
    # CONFIGURATION
    ##########################################################################
    ########
    # INPUTS
    ########
    train_ratio = 0.8
    mini_batch_ratios = [5,30,50,70,100]
    inference_file = '../raw_data/inferences/v1/proc_gt_users_images_rcnn.txt'
    #########
    # OUTPUTS
    #########
    data_out_file = '../out/train_test_split_rcnn_v1.json'
    ###########################################################################

    user_photos = retrieve_detected_objects(inference_file)
    train_test_info = train_test_split(user_photos, train_ratio, mini_batch_ratios)

    print('##############')
    print('#### INFO ####')
    print('##############')
    print('Number of valid users: ', len(train_test_info['users']))
    print('Numer of test users: ', len(list(train_test_info['test'].keys())))
    print('Train Info: ')
    for ratio, users in train_test_info['train'].items():
        print('\tratio = ',ratio,' nb_users = ',len(list(users.keys())))

    print('Saving ...')
    with open(data_out_file, 'w') as fp:
        json.dump(train_test_info,fp)
    print('Done!')

if __name__ == "__main__":
    main()