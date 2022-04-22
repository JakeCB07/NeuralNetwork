# Author: Jacob Burt

# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip.
import random
import numpy
import math
import sklearn.utils
from sklearn.utils import shuffle
import numpy as np
import nn_layer
import math_util as mu

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        new_layer = nn_layer.NeuralLayer(d, act)
        self.layers.append(new_layer)
        self.L += 1
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer

            for li in Layers[l0, l1...ln], a layer has 2 attributes. d and w.
            w.shape = (Layers[l-1].d+1) x (Layers[l].d) matrix
        '''
        np.random.seed()
        for l in range(1, self.L+1):
            self.layers[l].W = np.random.rand(self.layers[l-1].d +1, self.layers[l].d)
            self.layers[l].W = (self.layers[l].W * 2 -1)/math.sqrt(self.layers[l].d)

    
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.

        if(SGD):
            self.stochastic_GD(X, Y, eta, iterations, mini_batch_size)
        else:
            self.batch_GD(X, Y, eta, iterations)

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices.


    def batch_GD(self, X, Y, eta = 0.01, iterations = 1000):
        for i in range(iterations):
            n, d = X.shape
            self.forward_feed(X)
            self.layers[-1].Delta = (2 * (self.layers[-1].X[:,1:] - Y)) * (self.layers[-1].act_de(self.layers[-1].S))
            self.layers[-1].G = numpy.einsum('ij, ik-> jk', self.layers[-2].X,self.layers[-1].Delta) * (1 / n)
            self.back_propogation(X, n)

            for l in range(1, self.L+1):
                self.layers[l].W = self.layers[l].W - (eta*self.layers[l].G)

    def stochastic_GD(self,  X, Y, eta = 0.01, iterations = 10000, mini_batch_size = 1):
        N, d = X.shape
        batches = N // mini_batch_size
        X_shuffled, y_shuffled = sklearn.utils.shuffle(X, Y)
        for i in range(iterations):
            x_batch, y_batch = self.get_batches(X_shuffled, y_shuffled, mini_batch_size, batches, i)
            n, d = x_batch.shape
            self.forward_feed(x_batch)
            self.layers[-1].Delta = (2 * (self.layers[-1].X[:, 1:] - y_batch)) * (self.layers[-1].act_de(self.layers[-1].S))
            self.layers[-1].G = numpy.einsum('ij, ik-> jk', self.layers[-2].X, self.layers[-1].Delta) * (1 / (n+1))
            self.back_propogation(x_batch, n)
            for l in range(1, self.L+1):
                self.layers[l].W = (self.layers[l].W - (eta*self.layers[l].G))

    def get_batches(self, X, Y, mini_batch_size, batches, iteration):
        batch_start = ((iteration % batches) * mini_batch_size)
        batch_end = (((iteration % batches) + 1) * mini_batch_size -1)
        x_batch = X[batch_start:batch_end,:]
        y_batch = Y[batch_start:batch_end,:]

        return x_batch, y_batch

    def forward_feed(self, X):
        self.layers[0].X = np.insert(X, 0, 1, axis=1)
        for l in range(1, self.L+1):
            self.layers[l].S = self.layers[l-1].X @ self.layers[l].W
            self.layers[l].X = self.layers[l].act(np.insert(self.layers[l].S, 0, 1, axis=1))

        return self.layers[-1].X[:,1:]

    def back_propogation(self, X, n):
        for l in reversed(range(1, self.L)):
            self.layers[l].Delta = (self.layers[l].act_de(self.layers[l].S)) * \
                                   ((self.layers[l+1].Delta @ (np.transpose(self.layers[l+1].W)))[:,1:])
            self.layers[l].G = numpy.einsum('ij, ik-> jk', self.layers[l-1].X, self.layers[l].Delta) * (1/(n+1))

    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''
        x = self.forward_feed(X)
        return np.argmax(x, axis=1).reshape(-1,1)
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        x_labels = self.predict(X)
        y_labels = np.argmax(Y, axis=1).reshape(-1,1)
        n, d = x_labels.shape

        return np.sum(x_labels != y_labels) * (1/n)
 