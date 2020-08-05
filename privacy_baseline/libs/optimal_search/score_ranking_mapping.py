import numpy as np

def ranking(user_scores, pos_weigth = 0.5, print_ = False):
    """
    Ranking users in a situation

    :param user_scores:
         {user1: [pos, neg], ...}

    :param pos_weigth:
        positive weight (0,1)

    :return:
        user_ranking : dict
            {usr1: rank1, ...} -> max score -> best ranking

    """
    usr_list = list(user_scores.keys())
    pos_score = np.asarray([user_scores[usr][0] for usr in usr_list])
    neg_score = np.asarray([user_scores[usr][1] for usr in usr_list])

    if np.min(neg_score) < 0:
        neg_ranking = (neg_score + np.abs(np.min(neg_score)))/np.max(neg_score + np.abs(np.min(neg_score)))
    else:
        neg_ranking = neg_score

    if np.max(pos_score) > 0:
        pos_ranking = pos_score/np.max(pos_score)
    else:
        pos_ranking = pos_score

    scale_user_ranking = pos_weigth*pos_ranking + (1-pos_weigth)*neg_ranking

    descaled_user_ranking = list(np.round(scale_user_ranking*len(usr_list)).astype(int))

    if print_:
        print('     pos_score: ',pos_score)
        print('     neg_score: ',neg_score)

    user_ranking = {}
    for i, rank in enumerate(descaled_user_ranking):
        user_ranking[usr_list[i]] = rank

    return user_ranking

