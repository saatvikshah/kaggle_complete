import numpy as np
from transforms import *
from filters import *
from classifier import MLP
from sklearn.ensemble import RandomForestClassifier,\
    GradientBoostingClassifier,AdaBoostClassifier

#TODO: EOG based processing,ZCA Centering,Error Potential papers, Feature Reduction with PCA
#TODO: Possible use of time per session?,

"""
Model Feature Generators,Preprocessor,Classifiers
`get_preproc` : Single preprocessor which is applied when
loading raw data before any kind of preprocessing
(single file of single subject single session loaded then preprocessor applied
then divided into epochs by feedback event)
`proc_generator` : Apply chain of transforms to feedbackevent-epoched data(Only EEG features)
`clf_generator` : Apply a chain of estimators to features
"""


def get_preproc():
    return \
    TransformPipeline([BaseTransformer()])


def proc_generator():
    processors = [
        # [KeepChannel(tokeep=[29,30]),EEGConcatExtracter()],
        # [KeepChannel(tokeep=range(28,31)),ButterworthFilter(sampling_rate=200,fc1=0.1,fc2=15,input_type="epoched",order=4),EEGConcatExtracter(),Downsampler(method="dec",rate=8)],
        [KeepChannel(tokeep=[46]),ButterworthFilter(sampling_rate=200,fc1=0.1,fc2=20,input_type="epoched",order=6),EEGConcatExtracter(),Downsampler(method="dec",rate=4)],    #optimum
        # [KeepChannel(tokeep=[15,16,20] + range(21,53)),     #channels with 0.55+
        #  ButterworthFilter(sampling_rate=200,fc1=0.1,fc2=15,input_type="epoched",order=4),EEGConcatExtracter(),Downsampler(method="dec",rate=12)],
        # [KeepChannel(tokeep=[30,31,32,34,35,36,38,39,42,43,45,46,47,51,52]),     #channels with 0.57+
        #  ButterworthFilter(sampling_rate=200,fc1=0.1,fc2=20,input_type="epoched",order=4),EEGConcatExtracter(),Downsampler(method="dec",rate=12)],
        # [KeepChannel(tokeep=[34,39,46,51,52]),     #channels with 0.58+
        # ButterworthFilter(sampling_rate=200,fc1=0.1,fc2=20,input_type="epoched",order=6),EEGConcatExtracter(),Downsampler(method="dec",rate=30)],

    ]
    for processor in processors:
        yield TransformPipeline(processor)

def clf_generator():
    clfs = [
        # (RandomForestClassifier(random_state=42),
        # {'max_depth' : [3],'n_estimators' : [7000],'max_features' : [0.75],
        # 'min_samples_split' : [4]}),
        (GradientBoostingClassifier(),
        {'learning_rate' : [0.05],'n_estimators' : [500], 'max_depth' : [2],
        'max_features' : [0.25]}),  #Optimum
        # (MLP(),{}),
        # (LDA(),{}),
        # (SVC(),{'kernel' : ['linear'], 'C' : [1,2,3]}),
    ]
    for clf in clfs:
        yield clf[0],clf[1]