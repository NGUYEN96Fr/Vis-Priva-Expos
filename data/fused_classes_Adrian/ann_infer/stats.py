import os

def check_nb_img_per_class():
    """

    Returns
    -------

    """
    infer_class_dir = '/scratch_global/vankhoa/official_train_inference_ann_models/inference'
    classes = os.listdir(infer_class_dir)
    writer = open('infer_class_stats.txt','w')

    for class_ in classes:
        with open(os.path.join(infer_class_dir,class_)) as fp:

            lines = fp.readlines()
            writer.write('%s\\%s\n'%(class_.split('.')[0].replace('_',' '),len(lines)))

    writer.close()

def main():
    check_nb_img_per_class()


if __name__ == '__main__':
    main()