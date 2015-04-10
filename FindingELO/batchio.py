from pandas import DataFrame as df
from pandas.io.parsers import read_csv
import os
import numpy as np
import re
from extras import toint

def check_dir(dirpath):
    if not os.path.isdir(dirpath):
        raise("Directory : %s is missing" % dirpath)

def check_file(fpath):
    if not os.path.isfile(fpath):
        raise("File %s is missing" % str(fpath))

def parse_elofile(fpath):
    f = open(fpath,"r")
    everything = f.read()
    f.close()
    result_re = re.findall('\[Result \"(.+)\"\]', everything, re.IGNORECASE)
    whiteelo_re = re.findall('\[WhiteElo \"(\d+)\"\]', everything, re.IGNORECASE)
    blackelo_re = re.findall('\[BlackElo \"(\d+)\"\]', everything, re.IGNORECASE)
    train_set = len(whiteelo_re)
    test_set = len(result_re) - train_set
    whiteelo_re.extend(["None" for i in range(test_set)])
    blackelo_re.extend(["None" for i in range(test_set)])
    return result_re,whiteelo_re,blackelo_re

def clean_result(res):
    if res == "0-1":
        return -1
    elif res == "1-0":
        return 1
    elif res == "1/2-1/2":
        return 0


def data_parse(params,enginename = "stockfish",follow_uci=False):
    """
    Parses files generated from chess engines
    and loads relevant data into DataFrame
    """
    #Safety Checks
    check_dir(params.data_dir)
    elo_fpath = os.path.join(params.data_dir,"data%s.pgn" % ("_uci" if follow_uci else ""))
    results,whiteelos,blackelos = parse_elofile(elo_fpath)
    ce_fpath = os.path.join(params.data_dir,"%s.csv" % str(enginename))
    check_file(ce_fpath)
    data = read_csv(ce_fpath)
    data["Results"] = map(clean_result,results)
    data["WhiteEloScore"] = whiteelos
    data["BlackEloScore"] = blackelos
    data["MoveScores"] = data[["MoveScores"]].apply(lambda r: map(lambda x:map(toint,x),r.str.split(" ")))
    data["WhiteMoveScores"] = data["MoveScores"].apply(lambda x: [x[ind] for ind in range(len(x)) if ind % 2 == 1])
    data["BlackMoveScores"] = data["MoveScores"].apply(lambda x: [-x[ind] for ind in range(len(x)) if ind % 2 == 0])
    data["WhiteAdvantageMoveScores"] = data["MoveScores"].apply(lambda x:filter(lambda elem:elem >= 0,x))
    data["BlackAdvantageMoveScores"] = data["MoveScores"].apply(lambda x:filter(lambda elem:elem < 0,x))
    #Setup partitions
    for pt in range(params.partitions):
        data["Partition%dMoveScores" % pt] = data["MoveScores"].apply(lambda x:x[pt*len(x)/params.partitions : (pt+1)*len(x)/params.partitions])
        data["Partition%dWhiteMoveScores" % pt] = data["WhiteMoveScores"].apply(lambda x:x[pt*len(x)/params.partitions : (pt+1)*len(x)/params.partitions])
        data["Partition%dBlackMoveScores" % pt] = data["BlackMoveScores"].apply(lambda x:x[pt*len(x)/params.partitions : (pt+1)*len(x)/params.partitions])
        data["Partition%dWhiteAdvantageMoveScores" % pt] = data["WhiteAdvantageMoveScores"].apply(lambda x:x[pt*len(x)/params.partitions : (pt+1)*len(x)/params.partitions])
        data["Partition%dBlackAdvantageMoveScores" % pt] = data["BlackAdvantageMoveScores"].apply(lambda x:x[pt*len(x)/params.partitions : (pt+1)*len(x)/params.partitions])
    #Make slight modifications for better representation
    train_df = data[data.WhiteEloScore!="None"]
    test_df = data[data.WhiteEloScore=="None"].reset_index()
    return train_df,test_df

def create_submission(events,yw,yb):
    print "Preparing submission file...."
    submission = df()
    submission["Event"] = events
    submission["WhiteElo"] = yw
    submission["BlackElo"] = yb
    submission.to_csv("submission.csv",index=False)


