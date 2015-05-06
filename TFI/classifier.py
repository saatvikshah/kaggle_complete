from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,BaggingClassifier
from sklearn.lda import LDA
from sklearn.linear_model import Ridge,RidgeClassifier
from sklearn.svm import SVR,SVC
import pickle as pkl
import numpy as np
import os

class RandomForestRegressorWithCoef(RandomForestRegressor):
    def fit(self, *args, **kwargs):
        super(RandomForestRegressorWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class RidgeClassifierWithPredProba(RidgeClassifier):

    def predict_proba(self,X):
        d = self.decision_function(X)
        probs = np.exp(d) / np.sum(np.exp(d))
        return np.vstack((probs,1-probs)).T


class SimpleBlendedRegressor:

    def __init__(self,clf_list=[],clf_weights=[],feat_sel=[]):
        self.clf_list = clf_list
        self.feat_sel = feat_sel
        self.clf_weights = clf_weights

    def fit(self,X,y):
            for clf_idx in xrange(len(self.clf_list)):
                cols_idxs = self.get_cols(self.cols,self.feat_sel[clf_idx])
                Xtrain = X[:,cols_idxs]
                self.clf_list[clf_idx].fit(Xtrain,y)

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

class SimpleBlendedClassifier:

    def __init__(self,clf_list=[],clf_weights=[],feat_sel=[]):
        self.clf_list = clf_list
        self.feat_sel = feat_sel
        self.clf_weights = clf_weights

    def fit(self,X,y):
        self.num_labels = np.unique(y).shape[0]
        for clf_idx in xrange(len(self.clf_list)):
            cols_idxs = self.get_cols(self.cols,self.feat_sel[clf_idx])
            Xtrain = X[:,cols_idxs]
            self.clf_list[clf_idx].fit(Xtrain,y)

    def predict(self,X):
        sum_weights = sum(self.clf_weights)
        ypred = np.zeros((X.shape[0]))
        for clf_idx in xrange(len(self.clf_list)):
            cols_idxs = self.get_cols(self.cols,self.feat_sel[clf_idx])
            Xtest = X[:,cols_idxs]
            ypred += self.clf_list[clf_idx].predict(Xtest)*self.clf_weights[clf_idx]
        ypred = ypred/sum_weights
        return ypred

    def predict_proba(self,X):
        sum_weights = sum(self.clf_weights)
        ypred = np.zeros((X.shape[0],self.num_labels))
        for clf_idx in xrange(len(self.clf_list)):
            cols_idxs = self.get_cols(self.cols,self.feat_sel[clf_idx])
            Xtest = X[:,cols_idxs]
            ypred += self.clf_list[clf_idx].predict_proba(Xtest)*self.clf_weights[clf_idx]
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