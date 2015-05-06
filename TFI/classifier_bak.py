from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.lda import LDA
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import pylibfm
import pickle as pkl
import numpy as np
import os

def store_pickle(data,path):
        f = open(path,"w")
        pkl.dump(data,f)
        f.close()

def read_pickle(path):
        f = open(path,"r")
        data = pkl.load(f)
        return data

class RandomForestRegressorWithCoef(RandomForestRegressor):
    def fit(self, *args, **kwargs):
        super(RandomForestRegressorWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


class SimpleBlendedClassifier:

    def __init__(self,clf_list=[],clf_weights=[],feat_sel=[],cache=True):
        self.clf_list = clf_list
        self.feat_sel = feat_sel
        self.clf_weights = clf_weights
        self.cache = cache

    def fit(self,X,y):
        path = "./cache/%s" % str(self)
        if(os.path.isfile(path) and self.cache):
            self.clf_list = read_pickle(path)
        else:
            for clf_idx in xrange(len(self.clf_list)):
                cols_idxs = self.get_cols(self.cols,self.feat_sel[clf_idx])
                Xtrain = X[:,cols_idxs]
                path_clf = "./cache/%s" % str(self.clf_list[clf_idx]).translate(None,"(,)= ")
                if(os.path.isfile(path_clf) and self.cache):
                    self.clf_list[clf_idx] = read_pickle(path_clf)
                else:
                    self.clf_list[clf_idx].fit(Xtrain,y)
                    if self.cache:
                        store_pickle(self.clf_list[clf_idx],path_clf)
            if self.cache:
                store_pickle(self.clf_list,path)

    def predict(self,X):
        sum_weights = sum(self.clf_weights)
        ypred = np.zeros((X.shape[0]))
        for clf_idx in xrange(len(self.clf_list)):
            cols_idxs = self.get_cols(self.cols,self.feat_sel[clf_idx])
            Xtest = X[:,cols_idxs]
            ypred += self.clf_list[clf_idx].predict(Xtest)*self.clf_weights[clf_idx]
        ypred = ypred/sum_weights
        return ypred

    def add_all_colnames(self,colnames):
        self.cols = colnames

    def get_cols(self,cols,toselect):
        final = []
        for i,col in enumerate(cols):
            if col in toselect:
                final.append(i)
        return final

    def __repr__(self):
        base_name =  "SimpleBlendedRegModel"
        clf_name = '__'.join([str(name).translate(None,"(,)= ") for name in self.clf_list])
        clf_weights = '__'.join([str(wt) for wt in self.clf_weights])
        clf_featsel = '__'.join([''.join([colname for colname in col]) for col in self.feat_sel])
        return '___'.join([str(base_name),str(clf_name),clf_weights,clf_featsel])