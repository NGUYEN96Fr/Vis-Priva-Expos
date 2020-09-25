from loader.data_loader import train_test, gt_user_expos, vis_concepts

def data_loader(root, cfg):
    """

    :return:
    """
    X_batches, X_test_set = train_test(root, cfg.DATASETS.TRAIN_TEST_SPLIT)
    expos = gt_user_expos(root, cfg.DATASETS.GT_USER_EXPOS)
    concepts = vis_concepts(root, cfg.DATASETS.VIS_CONCEPTS)
    X_community = {}

    for user, objects in X_batches['100'].items():
        X_community[user] = objects
    for user, objects in X_test_set.items():
        X_community[user] = objects

    return X_batches, X_test_set, X_community, expos, concepts