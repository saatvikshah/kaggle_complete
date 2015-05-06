import json
import numpy as np

class Params:
    """
    Variable used to associate parameters derived from 'settings.json'
    Usage
    >>>  params = Params()
    >>> params.var1 = 100
    """
    pass


def load_params():
    """
    Load parameters from settings.json and store in Params object variable
    :return:set of parameters loaded from settings.json
    """
    json_data = open("settings.json")
    data = json.load(json_data)
    params = Params()
    params.data_dir = data["data_dir"]
    return params

def rmse(y_true,y_pred):
    assert (y_pred.shape[0] == y_true.shape[0]), "predicted and true values dont match in shape"
    num_samples = y_true.shape[0]
    return np.sqrt(np.sum(np.square(y_true - y_pred))/num_samples)