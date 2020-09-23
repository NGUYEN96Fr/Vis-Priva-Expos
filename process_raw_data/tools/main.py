import _init_paths
import os
import json
from data.train_test_split import train_test_split
from process.user_profiles import retrieve_detected_objects
from process.user_expos import load_gt_user_profiles, normalizing

def main():
    """
    
    """
    
    # configs
    ##########################################################################
    normalize = True
    inference_file = '../raw_data/inferences/v1/proc_gt_users_images_rcnn.txt'
    user_expo_paths = '../raw_data/user_exposures/v1'
    data_out_file = '../out/train_test_split_rcnn_v1.json'
    expo_out_file = '../out/gt_usr_exposure_v1.json'

    train_ratio = 0.8
    mini_batch_ratios = [5,30,50,70,100]

    ###########################################################################

    user_photos = retrieve_detected_objects(inference_file)
    train_test_info = train_test_split(user_photos, train_ratio, mini_batch_ratios)
    if normalize:
        norm_expo_paths = os.path.join(user_expo_paths, 'normalized_expos')
        if not os.path.exists(norm_expo_paths):
            os.makedirs(norm_expo_paths)
        normalizing(user_expo_paths, norm_expo_paths)
        user_score_situs = load_gt_user_profiles(norm_expo_paths, train_test_info['users'], normalize)

    else:
        user_score_situs = load_gt_user_profiles(user_expo_paths, train_test_info['users'], normalize)

    print('##############')
    print('#### INFO ####')
    print('##############')
    print('Number of valid users: ', len(train_test_info['users']))
    print('Numer of test users: ', len(list(train_test_info['test'].keys())))
    print('Train Info: ')
    for ratio, users in train_test_info['train'].items():
        print('\tratio = ',ratio,' nb_users = ',len(list(users.keys())))

    print('Situations: ')
    for situ, scores in user_score_situs.items():
        print('\t',situ.replace('_',' '),' nb_users = ',len(list(scores.keys())))

    print('Saving ...')
    with open(data_out_file, 'w') as fp:
        json.dump(train_test_info,fp)

    with open(expo_out_file, 'w') as fp1:
        json.dump(user_score_situs,fp1)
    print('Done!')

if __name__ == "__main__":
    main()