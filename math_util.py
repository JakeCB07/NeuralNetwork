#Author: Jacob Burt

import math

import numpy
import numpy as np

# Various math functions, including a collection of activation functions used in NN.

class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        return np.tanh(x)

    def _scalar_tanh_de(x):
        return 1-(MyMath.tanh(x) * MyMath.tanh(x))

    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        sum = MyMath.tanh(x) * MyMath.tanh(x) * -1
        return sum + 1

    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''
        v_logis = np.vectorize(MyMath._sigmoid)
        return v_logis(x)

    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''
        return 1 / (1+(math.e)**-s)

    def _scalar_logis_de(x):
        return MyMath._sigmoid(x) * (1-MyMath._sigmoid(x))

    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        sigmoid = MyMath.logis(x)
        return sigmoid * (1-sigmoid)

    def _scalar_iden(x):
        return x
    
    def iden(x):
        ''' Identity function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        return np.asarray(x)

    def _scalar_iden_de(x):
        return 1

    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        ret = MyMath.iden(x)
        return np.ones(ret.size)

    def _relu_scalar(x):
        return x * (x>0)

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        ret = MyMath.iden(x)
        with np.nditer(x, op_flags=['readwrite']) as thing:
            for i in thing:
                i[...] = max(0,i)

        return ret

    
    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.
        
            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        if x > 0:
            return 1
        return 0

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.   
        '''
        ret = MyMath.iden(x)
        v_relu_de = np.vectorize(MyMath._relu_de_scaler)
        return v_relu_de(ret)

    