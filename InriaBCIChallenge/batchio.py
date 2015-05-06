from pandas import read_csv
import os
from extras import *
import numpy as np
from model_gen import get_preproc

def read_metadata(params):
    """
    Return metadata about channels and extra associated information given in 'ChannelsLocation.csv'
    """
    metadata_fname = "ChannelsLocation.csv"
    return read_csv(os.path.join(params.data_dir,metadata_fname))

def read_maindata(params,type):
    """
    Read training/testing set files and extract the essential X,y matrices
    which can be processed further. To do this we first get list of files
    corresponding to subjects given in 'settings.json'. Then parse these to
    get a final list of dataframes with every item corresponding to a single
    feedback event and mapped y label
    :param params:'settings.json' paramters
    :param type:'train' or 'test'
    :return: if 'train' then returns X,y
             if 'test' then returns X
    """
    assert (type == "train" or type == "test"),"type field must be either train or test"
    flist,subjsess_list = get_filelist(params,type)
    X = []
    if type == "train":
        train_labels = read_csv(os.path.join(params.data_dir,"TrainLabels.csv"))
        y = []
        for findex in range(len(flist)):
            X.extend(get_x(read_csv(flist[findex]),0,260))
            y.extend(train_labels[
                train_labels["IdFeedBack"].str.contains(
                    subjsess_list[findex])]["Prediction"])
        assert (len(X) == len(y)),"Training and Prediction Set values dont match"
        return (np.array(X),y)
    elif type == "test":
        for findex in range(len(flist)):
            X.extend(get_x(read_csv(flist[findex]),0,260))
        return (np.array(X))

def get_x(my_df,start_offset,end_offset):
    """
    Every loaded csv file has EEG + EOG + FeedbackEvent info
    This function takes a loaded csv, extracts the necessary feedback events
    Finally depending on provided indices here it extracts required time periods
    of data for every feedback event
    :param my_df: csv read as dataframe
    :return: list of dataframes where each dataframe corresponds
    to data for a given time period corresponding to a FeedbackEvent
    """
    fb_list = []
    fb_indices = my_df[my_df["FeedBackEvent"] == 1].index.tolist()
    my_df = my_df.drop('FeedBackEvent', axis = 1).drop('Time',axis=1).drop('EOG',axis=1).as_matrix()
    # my_df = my_df.drop('FeedBackEvent', axis = 1).as_matrix()
    my_df = get_preproc().transform(my_df)
    for fb_ind in fb_indices:
        fb_list.append(my_df[fb_ind + start_offset:fb_ind + end_offset,:])
    return fb_list








