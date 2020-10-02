"""

The module studies interrater agreement of visual profile ratings

"""
import os
import numpy as np

class_encoding ={0: 'IT', 1: 'BANK',
                 2: 'WAIT', 3: 'ACCOM', 4: 'LIFI'}

path_ = '/home/nguyen/Documents/intern20/' \
       'Vis-Priva-Expos/process_raw_data/raw_data/user_exposures/v1/normalized_expos'

evaluators =  [ev for ev in os.listdir(path_) if '.txt' in ev]

situ_scores = {}

for evaluator in evaluators:
    eval_path = os.path.join(path_, evaluator)
    with open(eval_path) as fp:

        lines = fp.readlines()
        for line in lines:
            parts = line.split('\n')[0].split(' ')
            user = parts[0]
            situ = parts[1]
            score = float(parts[2])

            if situ not in situ_scores:
                situ_scores[situ] = {}
            if user not in situ_scores[situ]:
                situ_scores[situ][user] = []

            situ_scores[situ][user].append(score)

situ_hAD = [0,0,0,0,0]
situ_mAD = [0,0,0,0,0]

fix_threshold = 1.2

for situ, user_scores in situ_scores.items():
    ADs = []
    for user, scores in user_scores.items():
        scores = np.asarray(scores)
        mean  = np.mean(scores)
        residus = scores - mean
        abs_residus = np.abs(residus)
        AD_j = np.sum(abs_residus)/abs_residus.shape[0]
        if AD_j > fix_threshold:
            situ_hAD[int(situ)] = situ_hAD[int(situ)] + 1
        ADs.append(AD_j)

    situ_mAD[int(situ)] = np.mean(ADs)

for k in range(5):
    print("#----------------------------------#")
    print(class_encoding[k])
    print('mAD = ',situ_mAD[k])
    print('hAD= ',situ_hAD[k])