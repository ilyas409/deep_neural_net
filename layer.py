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
    def __init__(self, n_units, activation, weight_multiplier = 0.1):
        self.weight_multiplier = weight_multiplier
        self.n_units = n_units
        self.activation = activation
        
    def init_params(self, inp_size):
        np.random.seed(1)
        self.W = np.random.randn(self.n_units, inp_size) * self.weight_multiplier
        self.b = np.zeros((self.n_units, 1))
        
    def compute(self, X):
        self.Z = np.dot(self.W, X) + self.b
        self.A = self.activation.forward(self.Z)
        return self.A
    
    def __str__(self):
        return "Number of units:" + str(self.n_units) + ", " + str(self.activation)