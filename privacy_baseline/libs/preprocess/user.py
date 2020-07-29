import os
import numpy as np


def load_class_decoding():
    """

    Returns
    -------
        class_name_decoding : dict
            {'0': class1, ...}
    """
    return {0: 'job_search_IT', 1: 'bank_credit', 2: 'job_search_waiter_waitress', 3: 'accommodation_search'}



def load_gt_user_profiles(path):
    """
    Parameters
    ----------
    path : string
        path to the location of the crowd-scouring user profiles

    Returns
    -------
        user_score_situs: dict
            gt averaged privacy user scores per situation
                {situ1: {user1: avg_score, ...}, ...}
    """
    class_decoding = load_class_decoding()
    user_score_situs = {}

    annotators = [x for x in os.listdir(path) if '.md' not in x]

    for annotator in annotators:

        with open(os.path.join(path,annotator)) as fp:
            lines = fp.readlines()

            for line in lines:
                parts = line.split(' ')
                situ_name = class_decoding[int(parts[1])]

                if situ_name not in user_score_situs:
                    user_score_situs[situ_name] = {}
                user_ID = parts[0]

                if user_ID not in user_score_situs[situ_name]:
                    user_score_situs[situ_name][user_ID] = []

                score = float(parts[2]) - 4
                user_score_situs[situ_name][user_ID].append(score)


    for situ, users in user_score_situs.items():
        for user in users:
            user_score_situs[situ][user] = sum(user_score_situs[situ][user])/len(user_score_situs[situ][user])


    return user_score_situs