import json
from train_test_split import train_test_split
from process_usr_photos import retrieve_detected_objects
from process_crowdsourcing_usr_expo import load_gt_user_profiles

def main():
    """
    
    """
    
    ## configuration
    inference_file = 'raw_data/object_detection_on_user_photos/gt_profile_inferences_rcnn.txt'
    crowdsourcing_user_paths = 'raw_data/crowdsourcing_exposure_user_scores/'

    train_ratio = 0.8
    ratios = [30,50,70,100]

    user_photos = retrieve_detected_objects(inference_file)
    train_test_info = train_test_split(user_photos, train_ratio, ratios)
    user_score_situs = load_gt_user_profiles(crowdsourcing_user_paths, train_test_info['users'])

    print('##############')
    print('#### INFO ####')
    print('##############')
    print('Number of valid users: ',len(train_test_info['users']))
    print('Numer of test users: ',len(list(train_test_info['test'].keys())))
    print('Train Info: ')
    for ratio, users in train_test_info['train'].items():
        print('\tratio = ',ratio,' nb_users = ',len(list(users.keys())))
    print('Situations: ')
    for situ, scores in user_score_situs.items():
        print('\t',situ.replace('_',' '),' nb_users = ',len(list(scores.keys())))

    print('Saving ...')
    with open('out/train_test_split.json','w') as fp:
        json.dump(train_test_info,fp)

    with open('out/gt_usr_exposure.json','w') as fp1:
        json.dump(user_score_situs,fp1)
    print('Done!')

if __name__ == "__main__":
    main()