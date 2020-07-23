import os

def load_situs(path, denormalization = True):
    """Load object situation under a dictionary form

    :param path: string
        path to situations

    :return:
        class_situs : dict
            situation and its crowd-sourcing class exposure scores
                {situ1: {class1: score, ...}, ...}
    """
    class_situs = {}
    situs = os.listdir(path)


    for situ in situs:

        class_situs[situ] = {}
        with open(os.path.join(path, situ)) as fp:

            lines = fp.readlines()
            for line in lines:
                parts = line.split(' ')
                class_ = parts[0]
                if denormalization:
                    score = float(parts[1])*3
                else:
                    score = float(parts[1])

                class_situs[situ][class_] = score

    return class_situs