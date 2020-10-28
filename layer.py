import numpy as np

class Layer:
    """Abstract Layer class"""
    pass

class Input:
    """Input \"Layer\". Use it to specify the input size"""
    def __init__(self, input_size):
        self.input_size = input_size
    def __str__(self):
        return "Input size: " + str(self.input_size)
    
class Dense(Layer):
    """Densely-connected layer"""
    def __init__(self, n_units, activation, weight_multiplier=0.1, dropout=0):
        assert(weight_multiplier > 0)
        assert(dropout >= 0)
        assert(dropout < 1)
        self.weight_multiplier = weight_multiplier
        self.n_units = n_units
        self.activation = activation
        self.dropout = dropout
        self.D = None
        
    def init_params(self, inp_size):
        self.W = np.random.randn(self.n_units, inp_size) * self.weight_multiplier
        self.b = np.zeros((self.n_units, 1))
        
    def compute(self, X): 
        self.Z = np.dot(self.W, X) + self.b
        self.A = self.activation.forward(self.Z)
        if self.dropout > 0:
            D = np.random.rand(self.A.shape[0], self.A.shape[1])
            keep_prob = 1 - self.dropout
            self.D = (D < keep_prob).astype(int)
            self.A = self.A * self.D / keep_prob
        return self.A
    
    def __str__(self):
        return "Number of units:" + str(self.n_units) + ", " + str(self.activation)
    