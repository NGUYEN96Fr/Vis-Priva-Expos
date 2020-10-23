"""
The module analyses fine-tuning parameters


"""
import os
import pickle

def best_result(path, models, c, p):
    """
    Retrieve all values of the parameter and its best results

    Parameters
    ----------
    path: string
        path to models
    models:
        a set of trained models
    c: string
        component
    p: string
        component's param

    Returns
    -------
        dict:
            {param_val1: best1, ...}

    """
    result = {}

    for model_name in models:
        m_path = os.path.join(path, model_name)
        model = pickle.load(open(m_path,'rb'))
        model.set_seeds()
        model.test_vispel()
        m_result = model.test_result
        p_value = model.cfg[c][p]
        print(p_value)
