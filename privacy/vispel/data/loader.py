from loader.data_loader import train_test, gt_user_expos, vis_concepts

def data_loader(root, cfg):
    """

    :return:
    """
    batches, test_set = train_test(root, cfg.DATASETS.TRAIN_TEST_SPLIT)
    expos = gt_user_expos(root, cfg.DATASETS.GT_USER_EXPOS)
    concepts = vis_concepts(root, cfg.DATASETS.VIS_CONCEPTS)

    return batches, test_set, expos, concepts