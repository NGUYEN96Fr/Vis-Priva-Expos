
import os
import numpy as np

def normalizing(ini_expo_paths, save_path):
    """

    :param ini_expo_paths: string
    :param save_path: string

    :return:
    :references:
        https://kharshit.github.io/blog/2018/03/23/scaling-vs-normalization

    """
    annotators = [x for x in os.listdir(ini_expo_paths) if '.txt' in x]
    cls_encoding = load_class_decoding()

    for annotator in annotators:

        expo_by_situs = {}
        mean_std_by_situs = {}
        ann_path = os.path.join(ini_expo_paths,annotator)

        with open(ann_path) as fp:
            lines = fp.readlines()
            for line in lines:
                parts = line.split('\n')[0].split(' ')
                situ = parts[1]
                score = float(parts[2]) - 4
                if situ not in expo_by_situs:
                    expo_by_situs[situ] = []
                expo_by_situs[situ].append(score)

        for situ, scores in expo_by_situs.items():
            mean_std_by_situs[situ] = {}
            mean_std_by_situs[situ]['mean'] = np.mean(np.asarray(scores))
            mean_std_by_situs[situ]['std'] = np.std(np.asarray(scores))

        ann_save = os.path.join(save_path, annotator)
        writer = open(ann_save,'w')
        print('##################################')
        print(annotator.split('.')[0])
        for situ, mean_variance in mean_std_by_situs.items():
            print(cls_encoding[int(situ)],' mean=',mean_variance['mean'],' std=',mean_variance['std'])

        with open(ann_path) as fp:
            lines = fp.readlines()
            for line in lines:
                parts = line.split('\n')[0].split(' ')
                usr_id = parts[0]
                situ = parts[1]
                score = float(parts[2]) - 4
                norm_score = (score - mean_std_by_situs[situ]['mean'])/mean_std_by_situs[situ]['std']
                writer.write('%s %s %s'%(usr_id,situ,norm_score))
                writer.write('\n')

        writer.close()


def load_class_decoding():
    """

    Returns
    -------
        class_name_decoding : dict
            {'0': class1, ...}
    """
    return {0: 'job_search_IT', 1: 'bank_credit', 2: 'job_search_waiter_waitress', 3: 'accommodation_search', 4: 'life_insurance'}



def load_gt_user_profiles(path, valid_users, normalize):
    """
    Parameters
    ----------
        path : string
            path to the location of the crowd-scouring user profiles

    Returns
    -------
        user_score_situs: dict
            gt averaged privacy user corr per situation
                {situ1: {user1: avg_score, ...}, ...}

        valid_users: list

        normalize: boolean
            if normalize user exposures of each annotator

    """
    class_decoding = load_class_decoding()
    user_score_situs = {}

    annotators = [x for x in os.listdir(path) if '.txt' in x]

    for annotator in annotators:

        with open(os.path.join(path, annotator)) as fp:
            lines = fp.readlines()

            for line in lines:
                parts = line.split(' ')
                situ_name = class_decoding[int(parts[1])]

                if situ_name not in user_score_situs:
                    user_score_situs[situ_name] = {}
                user_ID = parts[0]

                if user_ID in valid_users:
                    if user_ID not in user_score_situs[situ_name]:
                        user_score_situs[situ_name][user_ID] = []

                    if normalize:
                        score = float(parts[2])
                    else:
                        score = float(parts[2]) - 4
                    user_score_situs[situ_name][user_ID].append(score)


    sel_usr_situs = {}
    for situ, users in user_score_situs.items():

        if situ != 'life_insurance':
            if situ not in sel_usr_situs:
                sel_usr_situs[situ] = {}
            for user in users:
                sel_usr_situs[situ][user] = sum(user_score_situs[situ][user])/len(user_score_situs[situ][user])


    return sel_usr_situs