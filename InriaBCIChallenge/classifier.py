import theano
import theano.tensor as T
import numpy as np
import time
from copy import deepcopy,copy

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

class Layer:
    """Neural Network Layer
    """

    def __init__(self, n_in, n_out,activation,hidden=False):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type activation: Tensor function
        :param activation: tensor function

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if hidden is True:
            rng = np.random.RandomState(1234)
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= W_values*4
        else:
            W_values = np.zeros((n_in, n_out),dtype=theano.config.floatX)
        self.W = theano.shared(
            value= W_values,
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.params = [self.W,self.b]
        self.activation = activation
        #self.ypred = T.argmax(self.decision_function_tensor(),axis=1)


    def decision_function_tensor(self,tensor_x):
        """
        Prediction function for Logistic Regression
        :param data_x: TensorType input of data
        :return:returns y predictions as a Tensor variable
        """
        lin_op = (T.dot(tensor_x, self.W) + self.b)
        if self.activation is None:
            return lin_op
        else:
            return self.activation(lin_op)

    def negative_log_likelihood(self, tensor_x, tensor_y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        tensor_ypredproba = self.decision_function_tensor(tensor_x)
        return -T.mean(T.log(tensor_ypredproba)[T.arange(tensor_y.shape[0]), tensor_y])

class LogisticRegression:
    """ Apply minibatch logistic regression

    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
                 which the datapoints lie

    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
                  which the labels lie

    """

    def __init__(self,n_in,n_out,batch_size=600,learning_rate=0.13,iterations=500,verbose=0):
        self.n_in = n_in
        self.n_out = n_out
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iters = iterations
        self.verbose = verbose

    def fit(self,data_x,data_y):
        print "Training"
        start = time.clock()
        n_batches = data_x.get_value(borrow=True).shape[0]/self.batch_size
        tensor_x = T.matrix('x')
        tensor_y = T.ivector('y')
        index = T.lscalar('index')
        self.single_layer = Layer(self.n_in,self.n_out,T.nnet.softmax)
        cost = self.single_layer.negative_log_likelihood(tensor_x, tensor_y)
        g_W = T.grad(cost,self.single_layer.W)
        g_b = T.grad(cost,self.single_layer.b)
        updates = [(self.single_layer.W,self.single_layer.W - g_W*self.learning_rate),
                    (self.single_layer.b,self.single_layer.b - g_b*self.learning_rate)]
        train_batch = theano.function([index],[cost],
                                      updates=updates,
                                      givens={tensor_x : data_x[index*self.batch_size : (index + 1)*self.batch_size],
                                              tensor_y : data_y[index*self.batch_size : (index + 1)*self.batch_size]})
        train_batch_costs = [0 for i in xrange(n_batches)]
        for iter in xrange(self.iters):
            for minibatch_index in xrange(n_batches):
                train_batch_costs[minibatch_index] = train_batch(minibatch_index)
            if self.verbose==1: print "Iter %d --> %f" % (iter,np.mean(train_batch_costs))
        end = time.clock()
        print "Finished Training Logistic Regression Model\n" \
              "Iterations %d\n" \
              "Time Taken : %d secs" % (self.iters,end - start)


    def predict(self,data_x):
        n_batches = data_x.get_value(borrow=True).shape[0]/self.batch_size
        tensor_x = T.matrix('x')
        index = T.lscalar('index')
        tensor_ypred = self.prediction_tensor(tensor_x)
        predictor = theano.function([index],tensor_ypred,
                                    givens={tensor_x : data_x[index*self.batch_size:(index + 1)*self.batch_size]})
        ypred = [predictor(i) for i in xrange(n_batches)]
        return np.hstack(ypred)

    def predict_proba(self,data_x):
        tensor_x = T.matrix('x')
        tensor_ypredproba = self.single_layer.decision_function_tensor(tensor_x)
        predproba_func = theano.function([],tensor_ypredproba,
                                           givens={tensor_x : data_x})
        return predproba_func()

    def prediction_tensor(self,data_x):
        """
        Returns the predicted y value as a tensor variable
        :param tensor_x: TensorType matrix on input data
        :return: TensorType tensor_ypred output
        """
        return T.argmax(self.single_layer.decision_function_tensor(data_x),axis=1)

    def accuracy(self, data_x,data_y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        n_batches = data_x.get_value(borrow=True).shape[0]/self.batch_size
        tensor_x = T.matrix('x')
        tensor_y  = T.ivector('y')
        index = T.lscalar('index')
        tensor_ypred = self.prediction_tensor(tensor_x)
        if tensor_y.ndim != tensor_ypred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', tensor_y.type, 'y_pred', tensor_ypred.type)
            )
        # check if y is of the correct datatype
        if tensor_y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            avg_acc_tensor = T.mean(T.eq(tensor_ypred, tensor_y))
        else:
            raise NotImplementedError()
        avg_accuracy_fn = theano.function([index],avg_acc_tensor,
                                       givens={tensor_x : data_x[index*self.batch_size:(index + 1)*self.batch_size],
                                               tensor_y : data_y[index*self.batch_size:(index + 1)*self.batch_size]})
        avg_acc = np.mean([avg_accuracy_fn(i) for i in xrange(n_batches)])
        return avg_acc

class MLP:

    def __init__(self,n_in,n_hidden,n_out,batch_size=600,learning_rate=0.13,iterations=200,l1=0.000,l2=0.0001,verbose=0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.hidden_layer = Layer(self.n_in,self.n_hidden,T.tanh,hidden=True)
        self.reg_layer = Layer(self.n_hidden,self.n_out,T.nnet.softmax)
        self.layers = [self.hidden_layer,self.reg_layer]
        self.L1 = sum(abs(layer.W).sum() for layer in self.layers)
        self.L2 = sum(abs(layer.W ** 2).sum() for layer in self.layers)
        self.params = []
        for layer in self.layers:   self.params.extend(layer.params)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iters = iterations
        self.verbose = verbose
        self.l1_factor = l1
        self.l2_factor = l2
        self.minibatch_count = 0
        self.train_err = []

    def decision_function_tensor(self,tensor_x):
        final_dec_fn = tensor_x
        for layer in self.layers:
            final_dec_fn = layer.decision_function_tensor(final_dec_fn)
        return final_dec_fn

    def negative_log_likelihood(self,tensor_x,tensor_y):
        tensor_ypredproba = self.decision_function_tensor(tensor_x)
        return -T.mean(T.log(tensor_ypredproba)[T.arange(tensor_y.shape[0]), tensor_y])

    def minibatch_trainer(self,data_x,data_y):
        tensor_x = T.matrix('x')
        tensor_y = T.ivector('y')
        cost = (
            self.negative_log_likelihood(tensor_x, tensor_y)
            +  self.l1_factor * self.L1
            + self.l2_factor * self.L2
        )
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        train_batch = theano.function([],[cost],
                                      updates=updates,
                                      givens={tensor_x : data_x,
                                              tensor_y : data_y})
        return train_batch()

    def partial_fit(self,data_x,data_y):
        self.minibatch_count += 1
        self.train_err.append(self.minibatch_trainer(data_x,data_y))
        if self.minibatch_count % 1000:
            if self.verbose==1: print "Iter %d --> %f" % (self.minibatch_count,np.mean(self.train_err))
            self.train_err = []

    def fit(self,data_x,data_y):
        #   Initializations
        tensor_x = T.matrix('x')
        tensor_y = T.ivector('y')
        index = T.lscalar('index')
        print "Training"
        start = time.clock()
        n_batches = data_x.get_value(borrow=True).shape[0]/self.batch_size
        cost = (
            self.negative_log_likelihood(tensor_x, tensor_y)
            +  self.l1_factor * self.L1
            + self.l2_factor * self.L2
        )
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        train_batch = theano.function([index],[cost],
                                      updates=updates,
                                      givens={tensor_x : data_x[index*self.batch_size : (index + 1)*self.batch_size],
                                              tensor_y : data_y[index*self.batch_size : (index + 1)*self.batch_size]})
        train_batch_costs = [0 for i in xrange(n_batches)]
        for iter in xrange(self.iters):
            for minibatch_index in xrange(n_batches):
                train_batch_costs[minibatch_index] = train_batch(minibatch_index)
            if self.verbose==1: print "Iter %d --> %f" % (iter,np.mean(train_batch_costs))
        end = time.clock()
        print "Finished Training Logistic Regression Model\n" \
              "Iterations %d\n" \
              "Time Taken : %d secs" % (self.iters,end - start)

    def predict(self,data_x):
        n_batches = data_x.get_value(borrow=True).shape[0]/self.batch_size
        tensor_x = T.matrix('x')
        index = T.lscalar('index')
        tensor_ypred = self.prediction_tensor(tensor_x)
        predictor = theano.function([index],tensor_ypred,
                                    givens={tensor_x : data_x[index*self.batch_size:
                                    (index + 1)*self.batch_size]})
        ypred = [predictor(i) for i in xrange(n_batches)]
        return np.hstack(ypred)

    def predict_proba(self,data_x):
        tensor_x = T.matrix('x')
        tensor_ypredproba = self.decision_function_tensor(tensor_x)
        predproba_func = theano.function([],tensor_ypredproba,
                                           givens={tensor_x : data_x})
        return predproba_func()

    def prediction_tensor(self,tensor_x):
        """
        Returns the predicted y value as a tensor variable
        :param tensor_x: TensorType matrix on input data
        :return: TensorType tensor_ypred output
        """
        return T.argmax(self.decision_function_tensor(tensor_x),axis=1)

    def accuracy(self, data_x,data_y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        n_batches = data_x.get_value(borrow=True).shape[0]/self.batch_size
        tensor_x = T.matrix('x')
        tensor_y  = T.ivector('y')
        index = T.lscalar('index')
        tensor_ypred = self.prediction_tensor(tensor_x)
        if tensor_y.ndim != tensor_ypred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', tensor_y.type, 'y_pred', tensor_ypred.type)
            )
        # check if y is of the correct datatype
        if tensor_y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            avg_acc_tensor = T.mean(T.eq(tensor_ypred, tensor_y))
        else:
            raise NotImplementedError()
        avg_accuracy_fn = theano.function([index],avg_acc_tensor,
                                       givens={tensor_x : data_x[index*self.batch_size:(index + 1)*self.batch_size],
                                               tensor_y : data_y[index*self.batch_size:(index + 1)*self.batch_size]})
        avg_acc = np.mean([avg_accuracy_fn(i) for i in xrange(n_batches)])
        return avg_acc

