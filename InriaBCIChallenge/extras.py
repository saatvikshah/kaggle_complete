import json
import os
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
    params.cache_dir = data["cache_dir"]
    params.train_subjects = data["train_subjects"]
    return params

def get_num_subjects(fnames):
    """
    Given the list of filenames(training/testing) it returns the list of subjects to which these files point
    :param fnames: list of filenames
    :return:list of assosciated subjects
    """
    fnames = [f.split("_")[1] for f in fnames]
    subjects = list(set([int(f.strip("S")) for f in fnames]))
    return subjects

def get_filelist(params,type):
    """
    Given the params variable and type of set required it returns only
    those list of files corresponding to number of subjects given for training input
    :param params: fixed Params containing values loaded from settings.json
    :param type: 'train' or 'test'
    :return: list of sorted files + list of associated subjects
    """
    assert (type == "train" or type == "test"),"type field must be either train or test"
    data_path = os.path.join(params.data_dir,type)
    if type == "train" : num_subjects = params.train_subjects
    if type == "test" : num_subjects = 10   #all subjects
    file_list = []
    subj_sess_names = []
    for dirname,_,fnames  in os.walk(data_path):
        subjects = get_num_subjects(fnames)[:num_subjects]
        subjects.sort()
        fnames.sort()
        for f in fnames:
            subj_num = int(f.split("_")[1].strip("S"))
            if subj_num in subjects:
                file_list.append(os.path.join(data_path,f))
                subj_sess_names.append(f.strip("Data_").strip(".csv"))
    return file_list,subj_sess_names

# def visualize_pretform(X,y,chan):
#     Xerr = []
#     Xnoerr = []
#     X = X[:,:,chan]
#     for i in xrange(len(y)):
#         if y[i] == 1:
#             Xerr.append(X[i])
#         else:
#             Xnoerr.append(X[i])
#     Xerravg = np.mean(np.vstack(Xerr),axis=0)
#     Xnoerravg = np.mean(np.vstack(Xnoerr),axis=0)
#     plt.plot(Xerravg,"--go",label="Error")
#     plt.plot(Xnoerravg,"--ro",label = "No Error")
#     plt.legend()
#     plt.show()

