import numpy as np
import math
from sklearn.preprocessing import scale
from scipy.signal import decimate
import scipy
from joblib import Parallel,delayed
import pickle as pkl
from extras import get_filelist
from pandas import read_csv,DataFrame
import pywt

"""
Transforms/Filters Pending
xDawn Filter
"""



class BaseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

"""
Specific Competition based features
"""

class EOGInfo:

    def __init__(self,data_type,params):
        assert (data_type == "train" or data_type == "test"),"data_type field must be either train or test"
        self.flist,self.subjsess_list = get_filelist(params,data_type)

    def transform(self):
        Xeogstats = []
        for findex in range(len(self.flist)):
            x_df = read_csv(self.flist[findex])
            fb_indices = x_df[x_df["FeedBackEvent"] == 1].index.tolist()
            for ind in fb_indices:
                xeogstats = []
                eogvals = x_df["EOG"].iloc[ind - 500:ind]
                xeogstats.append([eogvals.var(),eogvals.std()])
                Xeogstats.append(xeogstats)
        return np.vstack(Xeogstats)


class SubjectFeedbackInfo:

    def __init__(self,data_type,params):
        assert (data_type == "train" or data_type == "test"),"data_type field must be either train or test"
        self.flist,self.subjsess_list = get_filelist(params,data_type)

    def transform(self):
        subjectlist = map(lambda x:int(x[1:3]),self.subjsess_list)
        feedbacksesslist = map(lambda x:int(x[8:10]),self.subjsess_list)
        X = DataFrame()
        xsubj = []
        xsess = []
        xfeedbacknum = []
        xstartpos = []
        # xstartpostime = []    #time isnt really improving accuracy
        for findex in range(len(self.flist)):
            x_df = read_csv(self.flist[findex])
            fb_indices = x_df[x_df["FeedBackEvent"] == 1].index.tolist()
            # starttime_indices = x_df["Time"].iloc[fb_indices]
            del x_df
            fb_nums = range(len(fb_indices))
            subj_nums = [subjectlist[findex]]*len(fb_indices)
            sess_nums = [feedbacksesslist[findex]]*len(fb_indices)
            xsubj.extend(subj_nums)
            xsess.extend(sess_nums)
            xfeedbacknum.extend(fb_nums)
            xstartpos.extend(fb_indices)
            # xstartpostime.extend(starttime_indices)
        X["subject"] = xsubj
        X["sess"] = xsess
        X["feedback_num"] = xfeedbacknum
        X["start_pos"] = xstartpos
        # X["start_pos_time"] = xstartpostime
        return X.as_matrix()



class TransformPipeline:
    """
    Pure Transformer Pipeline
    """
    def __init__(self, transform_pipe):
        self.trans_pipe = transform_pipe

    def transform(self, X, y = None):
            for tform in self.trans_pipe:
                tform.fit(X,y)
                X = tform.transform(X)
            return X

    def __repr__(self):
        name = ""
        for tform in self.trans_pipe:
            name = '__'.join([name,str(tform)])
        return name

class FeatureConcatenate(BaseTransformer):

    """
        Feature Concatenate concatenates multiple transforms
        transform_list :
        a list for case of single transforms
        OR
        list of lists in case of concatenation of series of transforms
    """

    def __init__(self,transform_list):
        self.transform_list = transform_list
        self.tl_type = any(isinstance(el, list) for el in self.transform_list)

    def transform(self, X):
        if self.tl_type is True:    #list of lists in case of concatenation of series of transforms
            Xtform = []
            for t_list in self.transform_list:
                X_t = X
                for t_func in t_list:
                    t_func.fit(X_t)
                    X_t = t_func.transform(X_t)
                Xtform.append(X_t)
            return np.hstack(Xtform)
        else:                       #a list for case of single transforms
            Xtform = []
            for tform in self.transform_list:
                tform.fit(X)
                Xtform.append(tform.transform(X))
            return np.hstack(Xtform)

    def __repr__(self):
        name = ""
        if self.tl_type is False:
            for tform in self.transform_list:
                name = '__'.join([name,str(tform)])
        else:
            for t_list in self.transform_list:
                for tform in t_list:
                    name = '__'.join([name,str(tform)])
        return name

class EEGConcatExtracter(BaseTransformer):
    def __init__(self):
        pass

    def transform(self, X):
        n_epochs, n_samples, n_channels = X.shape
        X_tf = np.zeros((n_epochs, n_channels * n_samples))
        for epoch in xrange(n_epochs):
            for channel in xrange(n_channels):
                X_tf[epoch, channel * n_samples:(channel + 1) * n_samples] = X[epoch, :, channel]
        return X_tf

    def __repr__(self):
        return '_'.join(["EEGConcatExtracter"])

class Downsampler(BaseTransformer):
    """
    Downsample by
    -Averaging
    -Decimation
    """

    def __init__(self, method="avg", rate=10):
        """
        :param method:"avg" for averaging,"dec" for decimation
        :param rate: downsampling rate
        """
        self.method = method
        self.rate = rate

    def transform(self, X):
        if self.method == "avg":
            return self.averager(X)
        elif self.method == "dec":
            return self.decimater(X)

    def averager(self, X):
        n_epochs, n_features = X.shape
        Xavg = np.zeros((n_epochs, int(n_features / self.rate)))
        for ep in xrange(n_epochs):
            for feat in xrange(int(n_features / self.rate)):
                Xavg[ep, feat] = np.mean(X[ep, feat * self.rate:feat * self.rate + self.rate])
        return Xavg


    def decimater(self, X):
        n_epochs, n_features = X.shape
        X_ds = []
        for ep in xrange(n_epochs):
            X_ds.append(self.direct_downsampler(X[ep, :]))
        return np.array(X_ds)

    def direct_downsampler(self,Xep):
        num_features = Xep.shape[0]
        Xds = []
        for i in xrange(num_features):
            if i % self.rate == 0:
                Xds.append(Xep[i])
        return Xds

    def __repr__(self):
        return '_'.join(["Downsampler", str(self.method), str(self.rate)])

def apply_wt(X,cutoff_freq):
    n_samples, n_channels = X.shape
    b,a = scipy.signal.cheby1(4,0.5,cutoff_freq,btype='low')
    XWT2D = []
    for j in xrange(n_channels):
        Xchsamples = X[:,j]
        tformed=[]
        Xfilt = scipy.signal.filtfilt(b,a,Xchsamples)
        n_samples=len(Xfilt)
        while(n_samples>21):
            Xds = scipy.signal.resample(Xfilt,math.floor(n_samples/2))
            n_samples=len(Xds)
            b,a = scipy.signal.cheby1(4,0.5,cutoff_freq/2,btype='low')
            Xfiltds = scipy.signal.filtfilt(b,a,Xds)
            tformed.append(Xfiltds)
        XWT2D.append(np.hstack(tformed))
    return np.vstack(XWT2D).T

class WaveletTransform(BaseTransformer):

    #added by Ani

    def __init__(self,fs=247.0,fc=47.0):
        self.fs=fs
        self.fc=fc


    def transform(self,X):
        n_epochs, n_samples, n_channels = X.shape
        cutoff_freq = 2*self.fc/self.fs
        # Parallel Test
        XWT2d = np.array(Parallel(n_jobs=-1,verbose=2)(delayed(apply_wt)(X[k],cutoff_freq)
                                    for k in xrange(n_epochs)))
        #print(np.mean(XWT2d,axis=0))
        #mean=np.mean(XWT2d,axis=0)
        #print mean.shape
        return XWT2d

    def __repr__(self):
        return "WaveletTransform"

class ActualWaveletTransform(BaseTransformer):
    """
    Apply Wavelet transform on Epoched EEG Data
    Get the multilevel approx. coefficients
    """
    def __init__(self):
        self.wvlt = pywt.Wavelet('db2')

    def transform(self, X):
        n_epochs,n_samples,n_channels = X.shape
        print X.shape
        Xwvtf = []
        for epoch in xrange(n_epochs):
            Xepwvtf = []
            for chan in xrange(n_channels):
                Xepwvtf.append(pywt.dwt(X[epoch,:,chan],self.wvlt)[0])
            Xwvtf.append(Xepwvtf)
        Xwvtf = np.array(Xwvtf)
        n_epochs,n_channels,n_samples = Xwvtf.shape
        Xwvtf = Xwvtf.reshape(n_epochs,n_samples,n_channels)
        return Xwvtf


class RejectChannel(BaseTransformer):
    """
    Reject specific channels
    """

    def __init__(self, toreject, num_channels=64):
        all = range(num_channels)
        self.toreject = toreject
        self.tokeep = list(set(all) - set(toreject))


    def transform(self, X):
        return X[:, :, self.tokeep]

    def __repr__(self):
        return "RejectChannel_" + str(self.toreject)

class KeepChannel(BaseTransformer):
    """
    Reject specific channels
    """

    def __init__(self, tokeep):
        self.tokeep = tokeep


    def transform(self, X):
        num_epochs,num_samples,num_channels = X.shape
        return X[:, :, self.tokeep].reshape(num_epochs,num_samples,len(self.tokeep))

    def __repr__(self):
        return "KeepChannel_" + str(self.tokeep)

class FlattenedFFT(BaseTransformer):
    def __init__(self, slice_index):
        self.slice_index = slice_index

    def transform(self, X):
        n_epochs, n_samples, n_channels = X.shape
        Xfft = []
        for ep in xrange(n_epochs):
            xfft = []
            for chan in xrange(n_channels):
                xfft.append(np.log10(np.abs(np.fft.rfft(X[ep, :, chan], axis=0)[1:self.slice_index])))
            Xfft.append(np.hstack(xfft))
        return np.vstack(Xfft)

    def __repr__(self):
        return "FlattenedFFT_" + str(self.slice_index)

class FreqEigNCoeff(BaseTransformer):
    def __init__(self, slice_index):
        self.slice_index = slice_index

    def transform(self, X):
        n_epochs, n_samples, n_channels = X.shape
        Xfft = []
        for ep in xrange(n_epochs):
            xfft = []
            for chan in xrange(n_channels):
                xfft.append(np.log10(np.abs(np.fft.rfft(X[ep, :, chan], axis=0)[1:self.slice_index])))
            # Here samples per channel are features
            scaled = scale(np.vstack(xfft).T, axis=0)
            corr_matrix = np.corrcoef(scaled)
            eigenvalues = np.abs(np.linalg.eig(corr_matrix)[0])
            eigenvalues.sort()
            corr_coefficients = self.upper_right_triangle(corr_matrix)
            Xfft.append(np.concatenate((corr_coefficients, eigenvalues)))
        return np.vstack(Xfft)

    def upper_right_triangle(self, corr_matrix):
        num_d1, num_d2 = corr_matrix.shape
        coeff = []
        for i in xrange(num_d1):
            coeff.append(corr_matrix[i, i:])
        return np.hstack(coeff)

    def __repr__(self):
        return "FreqEigNCoeff_" + str(self.slice_index)

class TimeEigNCoeff(BaseTransformer):
    """
    Finds Time Eigenvalues and Flattened Correlation Matrix
    """

    def transform(self, X):
        n_epochs, n_samples, n_features = X.shape
        Xtime = []
        for ep in xrange(n_epochs):
            scaled = scale(np.vstack(X[ep].T), axis=0)
            corr_matrix = np.corrcoef(scaled)
            eigenvalues = np.abs(np.linalg.eig(corr_matrix)[0])
            eigenvalues.sort()
            corr_coefficients = self.upper_right_triangle(corr_matrix)
            Xtime.append(np.concatenate((corr_coefficients, eigenvalues)))
        return np.vstack(Xtime)

    def upper_right_triangle(self, corr_matrix):
        num_d1, num_d2 = corr_matrix.shape
        coeff = []
        for i in xrange(num_d1):
            coeff.append(corr_matrix[i, i:])
        return np.hstack(coeff)

    def __repr__(self):
        return "TimeEigNCoeff_"

class GlobalFeatures(BaseTransformer):
    """
    iterate through the matrix of 180*165*64(flashes,samples,channels)
    """

    def transform(self, X):
        n_epochs, n_samples, n_channels = X.shape
        Xop = []
        for i in range(n_epochs):
            output = []
            data_current = X[i]
            times = len(data_current)-1
            # delta1 = data_current[1:,:] - data_current[:times,:] # the 1st derivative
            # delta2 = delta1[1:,:] - delta1[:times-1,:] # the 2nd derivative
            output.append(np.max(data_current,axis=0))
            output.append(np.min(data_current,axis=0))
            output.append(np.var(data_current,axis=0))
            output.append(np.std(data_current,axis=0))
            output.append(np.mean(data_current,axis=0))
            output.append(np.median(data_current,axis=0))
            # output.append(np.max(np.max(np.absolute(data_current), axis=0)))
            # output.append(np.mean(np.max(np.absolute(data_current), axis=0)))
            # output.append(np.var(np.max(np.absolute(data_current), axis=0)))
            # output.append(np.var(np.max(data_current, axis=0)))
            # output.append(np.var(np.var(np.absolute(data_current), axis=0)))
            # output.append(np.mean(np.var(np.absolute(data_current), axis=0)))
            # # 1st Derivative Global Features
            # output.append(np.max(np.max(np.absolute(delta1), axis=0)))
            # output.append(np.mean(np.max(np.absolute(delta1), axis=0)))
            # output.append(np.var(np.max(np.absolute(delta1), axis=0)))
            # output.append(np.var(np.max(delta1, axis=0)))
            # output.append(np.var(np.var(np.absolute(delta1), axis=0)))
            # output.append(np.mean(np.var(np.absolute(delta1), axis=0)))
            # # 2nd Derivative Global Features
            # output.append(np.max(np.max(np.absolute(delta2), axis=0)))
            # output.append(np.mean(np.max(np.absolute(delta2), axis=0)))
            # output.append(np.var(np.max(np.absolute(delta2), axis=0)))
            # output.append(np.var(np.max(delta2, axis=0)))
            # output.append(np.var(np.var(np.absolute(delta2), axis=0)))
            # output.append(np.mean(np.var(np.absolute(delta2), axis=0)))
            Xop.append(np.hstack(output))
        return np.vstack(Xop)

    def __repr__(self):
        return "GlobalFeatures"

class xDAWN_filtering(BaseTransformer):

    def fit(self, X, y):
        if y is not None:
            X,D = self.generate_teoplitz_matrix(X,y)
            self.generate_filter(X,D)
        else:
            pass

    def transform(self, X):
        n_epochs,n_samples,n_channels=X.shape
        Xflat = X.reshape((n_epochs*n_samples,n_channels))
        filters = self.filters_load()
        projected_data = np.dot(Xflat,filters)
        del Xflat,filters
        """
        Now make the data into epochs
        """
        xDAWN=np.zeros((n_samples,n_channels))
        for i in range(n_epochs):
            xDAWN= np.dstack((xDAWN,projected_data[i*n_samples:(i+1)*n_samples,:]))
        return xDAWN[:,:,1:].T


    def generate_teoplitz_matrix(self,X,y):
        """
        label corresponding to the flashing(1 for target)
        run this code for all the data epochs
        """
        n_epochs,n_samples,n_channels = X.shape
        Xflat = X.reshape((n_epochs*n_samples,n_channels))
        for i in xrange(n_epochs):
            if y[i] == 1:
                D1 = np.diag(np.ones(n_samples))
            else:
                D1 = np.zeros((n_samples, n_samples))

            if i==0:
                D = D1
            else:
                D = np.vstack((D, D1))
        return Xflat,D


    def generate_filter(self, X ,D):

        """
        Compute QR factorisation
        """
        # QR decompositions of X and D
        Qx, Rx = np.linalg.qr(X)
            # QR decompositions of D
        Qd, Rd =  np.linalg.qr(D)

        """
        Compute SVD Qd.T Qx

        """
        Phi, Lambda, Psi = np.linalg.svd(np.dot(Qd.T, Qx),full_matrices=True)
        Psi = Psi.T
        #construct spatial filters
        for i in range(Psi.shape[1]):
            # Construct spatial filter with index i as Rx^-1*Psi_i
            ui = np.dot(np.linalg.inv(Rx), Psi[:,i])            #eq 12
            if i < Phi.shape[1]:
                ai = np.dot(np.dot(np.linalg.inv(Rd), Phi[:,i]),Lambda[i])   #eq 15
            if i == 0:
                filters = np.atleast_2d(ui).T
                ai = np.atleast_2d(ai)
            else:
                filters = np.hstack((filters,np.atleast_2d(ui).T))
                if i < Phi.shape[1]:
                    ai = np.vstack((ai, np.atleast_2d(ai)))
        self.filters_dump(filters)
        return filters

    def filters_dump(self,obj):
        f = open("filters.temp","wb")
        pkl.dump(obj,f)
        f.close()

    def filters_load(self):
        f = open("filters.temp","r")
        obj = pkl.load(f)
        f.close()
        return obj

    def __repr__(self):
        return "xDAWN_filtering"