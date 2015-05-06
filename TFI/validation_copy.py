from batchio import load_data,make_submission
from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV
import math
from extras import rmse
from classifier import *
import numpy as np
from sklearn.metrics import accuracy_score

selector = {
    "linear" : ['P8', 'P10', 'P12', 'P15', 'P17', 'P18', 'P22', 'P28', 'P32', 'num_months'],
    "lin_svr" : ['P8', 'P10', 'P12', 'P15', 'P16', 'P17', 'P18', 'P22', 'P24', 'P26', 'P28', 'P32', 'start_year', 'num_months'],
    "rf" : ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P19', 'P20', 'P21', 'P22', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P35', 'P36', 'P37', 'is_big_city', 'start_year', 'start_month', 'start_day', 'num_years', 'num_months', 'Type_Rest','rev_outlier']
}


def get_cols(cols,toselect):
    final = []
    for i,col in enumerate(cols):
        if col in toselect:
            final.append(i)
    return final


def gen_outlier_labels(revenue):
    mean = np.mean(revenue)
    std = np.std(revenue)
    labels = np.zeros(revenue.shape)
    upper_lim = mean + std
    lower_lim = mean - std
    for idx in xrange(revenue.shape[0]):
        yrev = revenue[idx]
        if yrev > upper_lim or yrev < lower_lim:
            labels[idx] = 1
        else:
            labels[idx] = 0
    return labels

def run_model_outliers(clf,data_type,data,scoring=accuracy_score):
    print "deriving outliers"
    assert(data_type == "train" or data_type == "test" or data_type == "cv")
    if data_type != "test":
        (X,y) = data
        if data_type == "train":
            print "training..."
            clf.fit(X,y)
            return clf
        else:
            print "validating..."
            num_cv_iters = 30
            score = []
            for iter in xrange(num_cv_iters):
                cv_iter = KFold(y.shape[0],n_folds=10,shuffle=True,random_state=iter)
                for train_idx,test_idx in cv_iter:
                    clf.fit(X[train_idx],y[train_idx])
                    score.append(scoring(y[test_idx],clf.predict(X[test_idx])))
            print "Mean : %f\nSTD: %f" % (np.mean(score),np.std(score))
    else:
        print "testing..."
        (X) = data
        return clf.predict(X)

def run_model(clf,data_type,data,scoring="mean_squared_error"):
    assert(data_type == "train" or data_type == "test" or data_type == "cv")
    if data_type != "test":
        (X,y) = data
        if data_type == "train":
            print "training..."
            # clf = BaggingRegressor(base_estimator=clf,n_estimators=1000,n_jobs=-1)
            clf.fit(X,np.log(y))
            return clf
        else:
            print "validating..."
            # cv_op = RFECV(clf,scoring=scoring,cv=137)
            # X = cv_op.fit_transform(X,np.log(y))
            # clf = BaggingRegressor(base_estimator=clf,n_estimators=1000,n_jobs=-1)
            num_cv_iters = 30
            loss = []
            for iter in xrange(num_cv_iters):
                cv_iter = KFold(y.shape[0],n_folds=10,shuffle=True,random_state=iter)
                for train_idx,test_idx in cv_iter:
                    clf.fit(X[train_idx],np.log(y[train_idx]))
                    print np.exp(clf.predict(X[test_idx]))
                    loss.append(rmse(y[test_idx],np.exp(clf.predict(X[test_idx]))))
            print "Log Loss Listing"
            print loss
            print "Mean : %f" % float(sum(loss)/len(loss))
            # return cv_op
    else:
        print "testing..."
        (X) = data
        return np.exp(clf.predict(X))

def get_predictions(clf,clf_out):
    print clf
    cols,Xtrain,ytrain = load_data("train")
    Xtest = load_data("test")
    cols = cols.tolist()
    clf_out.add_all_colnames(cols)
    yout = gen_outlier_labels(ytrain)
    # print selector
    # sel_idx = get_cols(cols,selector)
    # Xtrain = Xtrain[:,sel_idx]
    # Xtest = Xtest[:,sel_idx]
    print "getting outliers"
    # run_model_outliers(clf_out,"cv",(Xtrain,yout))
    clf_out = run_model_outliers(clf_out,"train",(Xtrain,yout))
    yout_pred = run_model_outliers(clf_out,"test",(Xtest))
    print "Running model"
    cols.append("rev_outlier")
    clf.add_all_colnames(cols)
    # run_model(clf,"cv",(Xtrain,ytrain))
    Xtrain = np.concatenate((Xtrain,yout.reshape(yout.shape[0],1)),axis=1)
    clf = run_model(clf,"train",(Xtrain,ytrain))
    Xtest = np.concatenate((Xtest,yout_pred.reshape(yout_pred.shape[0],1)),axis=1)
    ypred = run_model(clf,"test",(Xtest))
    return ypred

#Notes : 800-1200 estimators seems to work better tho giving a low validation score

if __name__ == '__main__':
    # np.random.seed(5)
    clf_blended = SimpleBlendedRegressor(
         clf_list=[
                  Ridge(),
                  # BaggingRegressor(SVR(C=0.6,degree=5,epsilon=0.2,gamma=0.09),n_estimators=1000,n_jobs=-1),
                  SVR(C=0.6,degree=5,epsilon=0.2,gamma=0.09),
                  ],
        clf_weights=[1,1],
        feat_sel=[
            selector["linear"],
            selector["rf"]
        ]
    )
    clf_outliers = SimpleBlendedClassifier(
         clf_list=[
                  # RidgeClassifier(),
                  RidgeClassifierWithPredProba(),
                  # BaggingRegressor(SVR(C=0.6,degree=5,epsilon=0.2,gamma=0.09),n_estimators=1000,n_jobs=-1),
                  # SVC(C=0.6,degree=5,gamma=0.09,probability=True),
                  ],
        clf_weights=[1],
        feat_sel=[
            selector["linear"],
            # selector["rf"]
        ]
    )
    ypred_blendedclf = get_predictions(clf_blended,clf_outliers)
    # ypred_ridge = get_predictions(Ridge(),"mean_squared_error",selector["linear"])
    # ypred_rf = get_predictions(RandomForestRegressor(n_estimators=1000,max_depth=5,min_samples_split=5,max_features=1,max_leaf_nodes=18,n_jobs=-1),"mean_squared_error",selector["rf"])
    # ypred_linsvr = get_predictions(LinearSVR(C=0.9,loss='squa red_epsilon_insensitive',epsilon=0.1),"mean_squared_error",selector["lin_svr"])
    # ypred_svr = get_predictions(SVR(C=0.6,degree=5,epsilon=0.2,gamma=0.09),"mean_squared_error",selector["rf"])
    # ypred_svrbagged = get_predictions(BaggingRegressor(SVR(C=0.6,degree=5,epsilon=0.2,gamma=0.09),n_estimators=1000,n_jobs=-1),"mean_squared_error",selector["rf"])
    # fm = pylibfm.FM(num_factors=10, num_iter=50,verbose=False,task="regression", initial_learning_rate=0.02, learning_rate_schedule="constant")
    # ypred_svrada = get_predictions(AdaBoostRegressor(SVR(C=0.6,degree=5,epsilon=0.2,gamma=0.09),n_estimators=20,learning_rate=0.5),"mean_squared_error",selector["rf"])
    # ypred = (ypred_ridge + ypred_svrbagged)/2
    make_submission("baggedsvrnridge.csv",ypred_blendedclf)
