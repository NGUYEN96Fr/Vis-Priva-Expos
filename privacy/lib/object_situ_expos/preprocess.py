
import os
import json


def preprocess(source, output):
    """Preprocess the object scores per situation files
    
    Parameters
    ----------
    source: string
        source directory containing the files

    output: string
        output path
    
    Outputs
    -------
        .txt file per annotator, divided by situations
    """

    ## dict of common object-situation
    common_objects = {}


    annotators = os.listdir(source)

    for annotator in annotators:
        file_ = os.path.join(source,annotator)

        with open(file_) as fp:
            situs = json.load(fp)

        situ_names = list(situs.keys())

