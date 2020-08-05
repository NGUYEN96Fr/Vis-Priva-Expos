import numpy as np
from numpy import linalg as LA

def unique_threshold(x_unique):
    """

    :param x_unique:
    :return:
    """
    count = 0
    sum_diff_norm2 = 0
    for i in range(x_unique.shape[0]):
        for j in range(i + 1,x_unique.shape[0]):
            count += 1
            sum_diff_norm2 += LA.norm(x_unique[i,:]-x_unique[j,:])

    sum_diff_norm2 = sum_diff_norm2 / count

    return  sum_diff_norm2

def detect_same_y_test(x_test, y_test, sum_diff_situ):
    """

    :param x_test:
    :param y_test:
    :param sum_situ:
    :return:
    """
    for i in range(x_test.shape[0]):
        for j in range(i+1, x_test.shape[0]):
            diff = LA.norm(x_test[i,:] - x_test[j,:])
            if diff < sum_diff_situ:
                print('val1: ',y_test[i],'  val2: ',y_test[j])
                print('diff = ',diff)


def train_test_observe(train_test_batch_situs):
    """

    :return:
    """
    print("##########################")
    for situ, data in train_test_batch_situs.items():
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(' ',situ)
        y_train = data['y_train']
        y_test = data['y_test']
        x_train = data['x_train']
        x_test = data['x_test']

        y_unique_train = list(np.unique(y_train))

        sum_diff_situ = 0
        count = 0

        for y_unique in y_unique_train:
            indexes = np.where(y_train == y_unique)[0]
            if indexes.shape[0] > 1:
                count += 1
                x_unique = x_train[indexes, :]
                sum_diff = unique_threshold(x_unique)
                sum_diff_situ += sum_diff

        sum_diff_situ = sum_diff_situ / count
        print('sum_diff_situ: ', sum_diff_situ)
        detect_same_y_test(x_test, y_test, sum_diff_situ)


    assert 1 == 2
