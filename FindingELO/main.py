from batchio import *
from extras import *
from features import *
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import warnings
import sys

def clf_generator():
    clfs = [
        # RandomForestRegressor(n_estimators=3000,verbose=1,n_jobs=1),
        # GradientBoostingRegressor(n_estimators=5000,verbose=1,learning_rate=0.01),
        LogisticRegression(),
    ]
    for clf in clfs:
        yield clf,clone(clf)

def main():
    params = load_params()
    train_df,test_df = data_parse(params)
    X,yw,yb = process_features(train_df,"train",params)
    del train_df
    #Scale up X
    for clf_w,clf_b in clf_generator():
        print list(X.columns.values)
        print clf_w
        # print cross_val_score(estimator=clf_w,
        #                       scoring="mean_absolute_error",
        #                       cv=4,
        #                       X=X,
        #                       y=yw,
        #                       n_jobs=-1)
        # print cross_val_score(estimator=clf_b,
        #                       scoring="mean_absolute_error",
        #                       cv=4,
        #                       X=X,
        #                       y=yb,
        #                       n_jobs=-1)
        # print "\n"
        # continue
        print "Training ..."
        clf_w.fit(X,yw)
        clf_b.fit(X,yb)
        print "Testing ..."
        X = process_features(test_df,"test",params)
        #print test_df.head()
        yw = clf_w.predict(X)
        yb = clf_b.predict(X)
        del X
        create_submission(test_df["Event"],yw,yb)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #sys.stdout = open("logs.txt","a")
    main()

