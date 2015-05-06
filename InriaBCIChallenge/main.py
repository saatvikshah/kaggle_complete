from batchio import *
from transforms import SubjectFeedbackInfo,EOGInfo
from extras import load_params
from sklearn.grid_search import GridSearchCV
from model_gen import proc_generator,clf_generator
import itertools
from classifier import shared_dataset

def cross_validation(X,y,clf,cv_params):
    cv_iterator = cv_indices(16,X.shape[0])
    clf_cv = GridSearchCV(clf,cv_params,cv=cv_iterator,scoring="roc_auc",n_jobs=-1,verbose=1)
    X,y = shared_dataset((X,y))
    clf_cv.fit(X,y)
    print "Cross Validation Stats"
    print("Best parameters set:")
    print(clf_cv.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf_cv.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    return clf_cv

def cv_indices(num_folds,num_samples):
    """
    Given number of samples and num_folds automatically create a subjectwise cross validator
    Assumption: per subject we have 340 samples of data
    >>> cv_set = cv_indices(2,680)
    >>> cv_set
    >>> (([0:340],[340:680]),([340:680,0:340]))
    Algo:
    1.Compute all the permutations.
    2.itreate through all the permutations and first calculate the train indices by taking first five then
    six,seven so on of each combination of arrangement.The rest will be the values of test indices
    3. Finally zip it to form the indices.
    :param num_folds: folds for cv
    :param num_samples: number of samples of input of data (should be a multiple of 340)
    :return: return a zipped list of tuples
     of ranges of training and testing data
    """
    n_epoch = 340
    n_subjects = num_samples/n_epoch
    rem=num_samples%n_epoch
    assert (rem == 0),"samples passed in not a multiple of 340"
    assert (num_folds<=n_subjects),"number of subjects is less then number of folds"
    n_set = np.round(n_subjects/num_folds)
    n_set = int(n_set)
    n_subjects=int(n_subjects)
    flag=[]
    for i in range(num_folds):
        if i<num_folds-1:
            flag=flag+[range(i*n_set,(i+1)*n_set)]
        else:
            flag=flag+[range(i*n_set,n_subjects)]
    train_indices=[]
    test_indices=[]
    #permutations=perm1(range(num_folds))
    permutations=list(itertools.combinations(range(num_folds),num_folds-1))
    permutations=map(list,permutations)
    sets = len(permutations)
    permutations_test=list(itertools.combinations(range(num_folds),1))
    permutations_test=map(list,permutations_test)
    permutations_test.reverse()
    for i in range(num_folds-1):
        for j in range(sets):
            for k in range(len(flag[permutations[j][i]])):
                if i<1:
                    train_indices=train_indices+[range(flag[permutations[j][i]][k]*n_epoch,(flag[permutations[j][i]][k]+1)*n_epoch)]
                    test_indices=test_indices+[range(flag[permutations_test[j][i]][k]*n_epoch,(flag[permutations_test[j][i]][k]+1)*n_epoch)]
                else:
                    train_indices=train_indices+[range(flag[permutations[j][i]][k]*n_epoch,(flag[permutations[j][i]][k]+1)*n_epoch)]
    custom_cv=zip(train_indices,test_indices)
    return custom_cv

def model_apply(data_type,clf_info,processor):
    params = load_params()
    if data_type == "train":
        print "Training"
        (X,y) = read_maindata(params,type="train")
        #visualize_pretform(X,y,19)
        (clf,cv_params) = clf_info
    if data_type == "test":
        print "Testing"
        (X) = read_maindata(params,type="test")
        (clf) = clf_info
    print processor
    Xeeg = processor.transform(X)
    Xfbinfo = SubjectFeedbackInfo(data_type,params).transform()
    # Xeog = EOGInfo(data_type,params).transform()
    X = np.concatenate((Xfbinfo,Xeeg),axis=1)
    del Xeeg,Xfbinfo
    print X.shape
    exit()
    if data_type == "train": return cross_validation(X,y,clf,cv_params)
    if data_type == "test": return clf.predict_proba(X)[:,1]

def main(mode):
    """
    main function in which the code is applied for either submission/
    model generation
    :param mode: "submission" or "cross_validation"
    :return:
    """
    if mode == "submission":
        params = load_params()
        ytest = read_csv(os.path.join(params.data_dir,"SampleSubmission.csv"))
        processor = proc_generator().next()
        clf = clf_generator().next()
        clf = model_apply("train",clf,processor)
        y_pred = model_apply("test",(clf),processor)
        ytest["Prediction"] = y_pred
        ytest.to_csv("submission_%s" % str(clf.best_estimator_).split("(")[0],index=False)
    elif mode == "cross_validation":
        for clf in clf_generator():
            for processor in proc_generator():
                model_apply("train",clf,processor)
    else:
        raise("Available modes are `submission` and `cross_validation`")

if __name__ == '__main__':
    main("cross_validation")

