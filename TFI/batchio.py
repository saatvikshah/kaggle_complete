from extras import load_params
from pandas import read_csv,DataFrame
import os
import numpy as np
from datetime import datetime

### Approaches
# 1. Using Tfidf Traanform : Give a weight to each class according to its value in a given example

def sort_restaurant_type(rest_type):
    if rest_type == "FC":
        return 0
    elif rest_type == "IL":
        return 1
    elif rest_type == "MB":
        return 1
    elif rest_type == "DT":
        return 1

def city_hash(cityname):
    return hash(cityname) % 57

def cal_count(given_date,mode="day"):
    current = "1/1/2015"
    start_date = datetime.strptime(given_date, "%m/%d/%Y")
    end_date = datetime.strptime(current, "%m/%d/%Y")
    req_val = (end_date-start_date)
    if mode == "day":
        return abs(req_val.days)
    elif mode == "month":
        return int(abs(req_val.days)/12)
    elif mode == "year":
        return abs(req_val.days)/365




def init_transforms(data):
    data["City"] = data["City"].apply(lambda x:str(x))
    data["is_big_city"] = data["City Group"].apply(lambda x:int(x=="Big Cities"))
    data["start_year"] = data["Open Date"].apply(lambda x:int(x[6:]))
    data["start_month"] = data["Open Date"].apply(lambda x:int(x[:2]))
    data["start_day"] = data["Open Date"].apply(lambda x:int(x[3:5]))
    data["num_years"] = data["start_year"].apply(lambda x:2015 - x)
    data["num_years_since_1900"] = data["start_year"].apply(lambda x:x - 1900)
    data["num_months"] = data["num_years"].apply(lambda x:x*12) - data["start_month"].apply(lambda x:12-x)
    data["num_months_since_1900"] = data["num_years_since_1900"].apply(lambda x:x*12) - data["start_month"].apply(lambda x:12-x)
    # data["Type_FC"] = data["Type"].apply(lambda x:int(x=="FC"))
    # data["Type_IL"] = data["Type"].apply(lambda x:int(x=="IL"))
    # data["Type_DT"] = data["Type"].apply(lambda x:int(x=="DT"))
    # data["Type_MB"] = data["Type"].apply(lambda x:int(x=="MB"))
    # New Ones
    data["Type_Rest"] = data["Type"].apply(sort_restaurant_type)
    data["City_Hash"] = data["City"].apply(city_hash)
    data = data.drop(["Id","City","City Group","Open Date",
                      "Type"],axis=1)
    # data = data[["num_months","is_big_city","P7"]]
    # Xd = data[['num_months','revenue']]
    # Xd["num_months"] = Xd["num_months"].apply(np.sqrt)
    # print Xd
    # data = data[["num_months"]["revenue"]]
    # print Xd.head()
    return data

def load_data(data_type,shuffle=True):
    assert(data_type == "train" or data_type == "test" or data_type == "cv")
    if data_type == "cv":   data_type = "train"
    path = os.path.join(load_params().data_dir,data_type + ".csv")
    data = read_csv(path)

    # print data.columns.values
    if data_type != "test":
        if shuffle: data = data.reindex(np.random.permutation(data.index))
        y = data.revenue.values
        data = data.drop(["revenue"],axis=1)
        data = init_transforms(data)
        cols = data.columns.values
        X = data.as_matrix()
        return cols,np.log(X + 1),y
    else:
        data = init_transforms(data)
        X = data.as_matrix()
        return np.log(X + 1)

def make_submission(name,preds):
    sub = read_csv(os.path.join(load_params().data_dir,'sampleSubmission.csv'))
    # create submission file
    sub['Prediction']=preds
    sub.to_csv(name, index=False)