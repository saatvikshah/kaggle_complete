import numpy as np
from scipy.stats import mode
from pandas import DataFrame as df
import pickle as hkl
import os

def process_features(my_df,type,params):
    try:
        if not os.path.isdir(params.cache_dir):
            os.makedirs(params.cache_dir)
        if type == "train":
            dict = hkl.load(open(str(os.path.join(params.cache_dir,"train_features.hkl")),"r"))
            return dict["X"],dict["y_white"],dict["y_black"]
        elif type == "test":
            dict = hkl.load(open(str(os.path.join(params.cache_dir,"test_features.hkl")),"r"))
            return dict["X"]
    except:
        X = df()
        #Global
        X["MaxMoveScore"] = my_df["MoveScores"].apply(np.max)
        X["MinMoveScore"] = my_df["MoveScores"].apply(np.min)
        X["RangeMoveScore"] = X["MaxMoveScore"] - X["MinMoveScore"]
        X["IQRMoveScore"] = my_df["MoveScores"].apply(safe_iqr)
        X["MedianMoveScore"] = my_df["MoveScores"].apply(np.median)
        X["STDMoveScore"] = my_df["MoveScores"].apply(np.std)
        X["GameLen"] = my_df["MoveScores"].apply(len)
        X["MeanMoveScore"] = my_df["MoveScores"].apply(np.mean)
        X["ModeMoveScore"] = my_df["MoveScores"].apply(mode,axis=0).apply(lambda x:x[0][0])
        # X["ModeRound10MoveScore"] = my_df["MoveScores"].apply(lambda x:np.round(x,-1)).apply(mode,axis=0).apply(lambda x:x[0][0])
        X["SumMoveScore"] = my_df["MoveScores"].apply(np.sum)
        X["BlunderCount"] = my_df["MoveScores"].apply(catch_blunders)
        X["Results"] = my_df["Results"]
        # White Scores
        X["WhiteMaxMoveScore"] = my_df["WhiteMoveScores"].apply(np.max)
        X["WhiteMinMoveScore"] = my_df["WhiteMoveScores"].apply(np.min)
        X["WhiteRangeMoveScore"] = X["WhiteMaxMoveScore"] - X["WhiteMinMoveScore"]
        X["WhiteIQRMoveScore"] = my_df["WhiteMoveScores"].apply(safe_iqr)
        X["WhiteMedianMoveScore"] = my_df["WhiteMoveScores"].apply(safe_median)
        X["WhiteSTDMoveScore"] = my_df["WhiteMoveScores"].apply(safe_std)
        X["WhiteMeanMoveScore"] = my_df["WhiteMoveScores"].apply(safe_mean)
        X["WhiteModeMoveScore"] = my_df["WhiteMoveScores"].apply(safe_mode).apply(lambda x:x[0][0])
        X["WhiteSumMoveScore"] = my_df["WhiteMoveScores"].apply(np.sum)
        X["WhiteBlunderCount"] = my_df["WhiteMoveScores"].apply(catch_blunders)
        # Black Scores
        X["BlackMaxMoveScore"] = my_df["BlackMoveScores"].apply(np.max)
        X["BlackMinMoveScore"] = my_df["BlackMoveScores"].apply(np.min)
        X["BlackRangeMoveScore"] = X["BlackMaxMoveScore"] - X["BlackMinMoveScore"]
        X["BlackIQRMoveScore"] = my_df["BlackMoveScores"].apply(safe_iqr)
        X["BlackMedianMoveScore"] = my_df["BlackMoveScores"].apply(safe_median)
        X["BlackSTDMoveScore"] = my_df["BlackMoveScores"].apply(safe_std)
        X["BlackMeanMoveScore"] = my_df["BlackMoveScores"].apply(safe_mean)
        X["BlackModeMoveScore"] = my_df["BlackMoveScores"].apply(safe_mode).apply(lambda x:x[0][0])
        X["BlackSumMoveScore"] = my_df["BlackMoveScores"].apply(np.sum)
        X["BlackBlunderCount"] = my_df["BlackMoveScores"].apply(catch_blunders)
        #White Advantage
        #X["WhiteAdvantageMaxMoveScore"] = my_df["MoveScores"].apply(lambda x:filter(lambda elem:elem >= 0,x)).apply(safe_max)  #Useless
        X["WhiteAdvantageMinMoveScore"] = my_df["WhiteAdvantageMoveScores"].apply(safe_min)
        X["WhiteAdvantageRangeMoveScore"] = X["MaxMoveScore"] - X["WhiteAdvantageMinMoveScore"]
        X["WhiteAdvantageIQRMoveScore"] = my_df["WhiteAdvantageMoveScores"].apply(safe_iqr)
        X["WhiteAdvantageMedianMoveScore"] = my_df["WhiteAdvantageMoveScores"].apply(safe_median)
        X["WhiteAdvantageSTDMoveScore"] = my_df["WhiteAdvantageMoveScores"].apply(safe_std)
        X["WhiteAdvantageCount"] = my_df["WhiteAdvantageMoveScores"].apply(len)
        X["WhiteAdvantageMeanMoveScore"] = my_df["WhiteAdvantageMoveScores"].apply(safe_mean)
        X["WhiteAdvantageModeMoveScore"] = my_df["WhiteAdvantageMoveScores"].apply(safe_mode).apply(lambda x:x[0][0])
        # X["WhiteAdvantageModeRound10MoveScore"] = my_df["MoveMoveScores"].apply(lambda x:filter(lambda elem:elem >= 0,x)).apply(lambda x:np.round(x,-1)).apply(safe_mode).apply(lambda x:x[0][0])

        #Black Advantage
        X["BlackAdvantageMaxMoveScore"] = my_df["BlackAdvantageMoveScores"].apply(safe_max)
        #X["BlackAdvantageMinMoveScore"] = my_df["MoveMoveScores"].apply(lambda x:filter(lambda elem:elem < 0,x)).apply(safe_min)   #Useless
        X["BlackAdvantageRangeMoveScore"] = X["BlackAdvantageMaxMoveScore"] - X["MinMoveScore"]
        X["BlackAdvantageIQRMoveScore"] = my_df["BlackAdvantageMoveScores"].apply(safe_iqr)
        X["BlackAdvantageMedianMoveScore"] = my_df["BlackAdvantageMoveScores"].apply(safe_median)
        X["BlackAdvantageSTDMoveScore"] = my_df["BlackAdvantageMoveScores"].apply(safe_std)
        X["BlackAdvantageCount"] = my_df["BlackAdvantageMoveScores"].apply(len)
        X["BlackAdvantageMeanMoveScore"] = my_df["BlackAdvantageMoveScores"].apply(safe_mean)
        X["BlackAdvantageModeMoveScore"] = my_df["BlackAdvantageMoveScores"].apply(safe_mode).apply(lambda x:x[0][0])
        # X["BlackAdvantageModeRound10MoveScore"] = my_df["BlackAdvantageScores"].apply(lambda x:np.round(x,-1)).apply(safe_mode).apply(lambda x:x[0][0])

        #Partitioning

        ## All moves
        X["AllMoveScoresPartitionLen"] = my_df["Partition0MoveScores"].apply(len)
        for pt in range(params.partitions):
            X["Partition%dMaxMoveScore" % pt] = my_df["Partition%dMoveScores" % pt].apply(safe_max)
            X["Partition%dMinMoveScore" % pt] = my_df["Partition%dMoveScores" % pt].apply(safe_min)
            X["Partition%dRangeMoveScore" % pt] = X["Partition%dMaxMoveScore" % pt] - X["Partition%dMinMoveScore" % pt]
            X["Partition%dIQRMoveScore" % pt] = my_df["Partition%dMoveScores" % pt].apply(safe_iqr)
            X["Partition%dMedianMoveScore" % pt] = my_df["Partition%dMoveScores" % pt].apply(safe_median)
            X["Partition%dSTDMoveScore" % pt] = my_df["Partition%dMoveScores" % pt].apply(safe_std)
            X["Partition%dMeanMoveScore" % pt] = my_df["Partition%dMoveScores" % pt].apply(safe_mean)
            X["Partition%dModeMoveScore" % pt] = my_df["Partition%dMoveScores" % pt].apply(safe_mode).apply(lambda x:x[0][0])
            X["Partition%dSumMoveScore" % pt] = my_df["Partition%dMoveScores" % pt].apply(np.sum)
            X["Partition%dBlunderCount" % pt] = my_df["Partition%dMoveScores" % pt].apply(catch_blunders)

        ## White moves
        X["WhiteMoveScoresPartitionLen"] = my_df["Partition0WhiteMoveScores"].apply(len)
        for pt in range(params.partitions):
            X["Partition%dMaxWhiteMoveScore" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(safe_max)
            X["Partition%dMinWhiteMoveScore" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(safe_min)
            X["Partition%dRangeWhiteMoveScore" % pt] = X["Partition%dMaxWhiteMoveScore" % pt] - X["Partition%dMinWhiteMoveScore" % pt]
            X["Partition%dIQRWhiteMoveScore" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(safe_iqr)
            X["Partition%dMedianWhiteMoveScore" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(safe_median)
            X["Partition%dSTDWhiteMoveScore" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(safe_std)
            X["Partition%dMeanWhiteMoveScore" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(safe_mean)
            X["Partition%dModeWhiteMoveScore" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(safe_mode).apply(lambda x:x[0][0])
            X["Partition%dSumWhiteMoveScore" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(np.sum)
            X["Partition%dWhiteBlunderCount" % pt] = my_df["Partition%dWhiteMoveScores" % pt].apply(catch_blunders)


        ## Black moves
        X["BlackMoveScoresPartitionLen"] = my_df["Partition0BlackMoveScores"].apply(len)
        for pt in range(params.partitions):
            X["Partition%dMaxBlackMoveScore" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(safe_max)
            X["Partition%dMinBlackMoveScore" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(safe_min)
            X["Partition%dRangeBlackMoveScore" % pt] = X["Partition%dMaxBlackMoveScore" % pt] - X["Partition%dMinBlackMoveScore" % pt]
            X["Partition%dIQRBlackMoveScore" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(safe_iqr)
            X["Partition%dMedianBlackMoveScore" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(safe_median)
            X["Partition%dSTDBlackMoveScore" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(safe_std)
            X["Partition%dMeanBlackMoveScore" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(safe_mean)
            X["Partition%dModeBlackMoveScore" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(safe_mode).apply(lambda x:x[0][0])
            X["Partition%dSumBlackMoveScore" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(np.sum)
            X["Partition%dBlackBlunderCount" % pt] = my_df["Partition%dBlackMoveScores" % pt].apply(catch_blunders)

        ## WhiteAdvantage moves
        X["WhiteAdvantageMoveScoresPartitionLen"] = my_df["Partition0WhiteAdvantageMoveScores"].apply(len)
        for pt in range(params.partitions):
            X["Partition%dMaxWhiteAdvantageMoveScore" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(safe_max)
            X["Partition%dMinWhiteAdvantageMoveScore" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(safe_min)
            X["Partition%dRangeWhiteAdvantageMoveScore" % pt] = X["Partition%dMaxWhiteAdvantageMoveScore" % pt] - X["Partition%dMinWhiteAdvantageMoveScore" % pt]
            X["Partition%dIQRWhiteAdvantageMoveScore" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(safe_iqr)
            X["Partition%dMedianWhiteAdvantageMoveScore" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(safe_median)
            X["Partition%dSTDWhiteAdvantageMoveScore" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(safe_std)
            X["Partition%dMeanWhiteAdvantageMoveScore" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(safe_mean)
            X["Partition%dModeWhiteAdvantageMoveScore" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(safe_mode).apply(lambda x:x[0][0])
            X["Partition%dSumWhiteAdvantageMoveScore" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(np.sum)
            X["Partition%dWhiteAdvantageBlunderCount" % pt] = my_df["Partition%dWhiteAdvantageMoveScores" % pt].apply(catch_blunders)
            
        ## BlackAdvantage moves
        X["BlackAdvantageMoveScoresPartitionLen"] = my_df["Partition0BlackAdvantageMoveScores"].apply(len)
        for pt in range(params.partitions):
            X["Partition%dMaxBlackAdvantageMoveScore" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(safe_max)
            X["Partition%dMinBlackAdvantageMoveScore" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(safe_min)
            X["Partition%dRangeBlackAdvantageMoveScore" % pt] = X["Partition%dMaxBlackAdvantageMoveScore" % pt] - X["Partition%dMinBlackAdvantageMoveScore" % pt]
            X["Partition%dIQRBlackAdvantageMoveScore" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(safe_iqr)
            X["Partition%dMedianBlackAdvantageMoveScore" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(safe_median)
            X["Partition%dSTDBlackAdvantageMoveScore" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(safe_std)
            X["Partition%dMeanBlackAdvantageMoveScore" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(safe_mean)
            X["Partition%dModeBlackAdvantageMoveScore" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(safe_mode).apply(lambda x:x[0][0])
            X["Partition%dSumBlackAdvantageMoveScore" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(np.sum)
            X["Partition%dBlackAdvantageBlunderCount" % pt] = my_df["Partition%dBlackAdvantageMoveScores" % pt].apply(catch_blunders)


        if type == "train":
            y_white = my_df["WhiteEloScore"].apply(int)
            y_black = my_df["BlackEloScore"].apply(int)
            dict = {}
            dict["X"] = X
            dict["y_white"] = y_white
            dict["y_black"] = y_black
            hkl.dump(dict,open(str(os.path.join(params.cache_dir,"train_features.hkl")),"wb"))
            return X,y_white,y_black
        elif type == "test":
            dict = {}
            dict["X"] = X
            hkl.dump(dict,open(str(os.path.join(params.cache_dir,"test_features.hkl")),"wb"))
            return X


def catch_blunders(cp_list):
    blunder_count = {}
    count = 0
    for ind in xrange(len(cp_list)):
        if ind > 0:
            if cp_list[ind] < cp_list[ind - 1]:
                count += 1
            else:
                if str(count) in blunder_count.keys():
                    blunder_count[str(count)] += 1
                else:
                    blunder_count[str(count)] = 1
                count = 0
    summer = 0
    for key in blunder_count.keys():
        summer += blunder_count[key]*int(key)
    return summer

def safe_max(cp_list):
    if len(cp_list) == 0:
        return 0
    else:
        return np.max(cp_list)

def safe_min(cp_list):
    if len(cp_list) == 0:
        return 0
    else:
        return np.min(cp_list)

def safe_median(cp_list):
    if len(cp_list) == 0:
        return 0
    else:
        return np.median(cp_list)

def safe_std(cp_list):
    if len(cp_list) == 0:
        return 0
    else:
        return np.std(cp_list)

def safe_mean(cp_list):
    if len(cp_list) == 0:
        return 0
    else:
        return np.mean(cp_list)

def safe_mode(cp_list):
    if len(cp_list) == 0:
        return [[0]]
    else:
        return mode(cp_list)

def safe_iqr(cp_list):
    if len(cp_list) == 0:
        return 0
    else:
        return np.subtract(*np.percentile(cp_list, [75, 25]))
