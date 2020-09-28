import _init_paths
import os
import json

def user_scores():
    """
    Statistics on the crowd-sourcing user corr

    :return:

    """
    path = '../out/gt_usr_exposure_v1.json'
    writer =  open('score_stats.txt','w')

    with open(path,'rb') as json_file:
        score_situs = json.load(json_file)

    for situ, scores in score_situs.items():
        score_groups = []
        for user, score in scores.items():
            if score not in score_groups:
                score_groups.append(score)
        writer.write('%s:   %s score groups'%(situ,len(score_groups)))
        writer.write('\n')

    writer.close()


def main():
    user_scores()

if __name__ == '__main__':
    main()