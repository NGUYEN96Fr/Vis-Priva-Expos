import os
import sys
import  configparser
import _init_paths
from user_imgs.retrieve import retrieve_detected_objects, retrieve_photos
from situations.load_situs import load_situs
from user_situ_expos.user_expo import _photos_users
from detectors.active import active_detectors
from focal_exposure.focal_exposure import focal_exposure

def main():
    ##get root directory
    root = os.path.dirname(os.getcwd())

    ##read the param config file
    conf = configparser.ConfigParser()
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]

    ##get params
    inference_file = conf['inference_file']
    siutation_file = conf['situation_file']
    f_top = float(conf['f_top'])
    K = float(conf['K'])
    gamma = float(conf['gamma'])

    ##Read user's photos
    objects_photo_per_user = retrieve_detected_objects(os.path.join(root, inference_file))

    ##Read object exposures in each situation
    object_expo_situs = load_situs(os.path.join(root, siutation_file))

    ##Estimate exposure of user's photos in each situation
    print('Estimate exposure user photos ...')
    user_photo_expo_situs = {}
    for situ_name, expo_clss in object_expo_situs.items():
        #activated detectors
        detectors = active_detectors(expo_clss)
        #estimate user's photo exposure
        user_photo_expo_situs[situ_name.split('.')[0]] = _photos_users(objects_photo_per_user,f_top,detectors)
    print('Done!')

    print('Study Focal Exposure Impact ...')
    ## STUDY OF FOCAL EXPOSURE IMPACT ON PHOTO EXPOSURE HISTOGRAM IN EACH SITUATION
    gamma_vals = [0,1,2,3,4]
    scaled_exposure_situs = {}
    for situ_name, users in user_photo_expo_situs.items():
        scaled_exposure_situs[situ_name] = {}

        for user, photos in users.items():
            for photo, score_obj in photos.items():
                for gamma_ in gamma_vals:
                    if gamma_ not in scaled_exposure_situs[situ_name]:
                        scaled_exposure_situs[situ_name][gamma_] = []
                    scaled_exposure_situs[situ_name][gamma_].append(focal_exposure(score_obj[0],gamma_,K))

    print('Done!')

if __name__ == '__main__':
    main()