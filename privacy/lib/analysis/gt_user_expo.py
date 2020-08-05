import numpy as np


def mean_variance(gt_expo_situs):
     """

     :return:
     """
     for situ, users in gt_expo_situs.items():
          print('   ',situ)
          scores = [users[user] for user in list(users.keys())]

          print('mean = ', np.mean(scores))
          print('variance = ', np.var(scores))

     assert 1 == 2