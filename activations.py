import numpy as np
from enum import Enum

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

class Activation:
    def __init__(self, activation):
        self.activation = activation
    def __str__(self):
        return str(self.activation)
    def forward(self, x):
        if self.activation == Activations.ReLU:
            return relu(x)
        elif self.activation == Activations.Sigmoid:
            return sigmoid(x)
        else:
            raise Exception("Unknown activation function")
    def backward(self, dA, x):
        if self.activation == Activations.ReLU:
            return relu_backward(dA, x)
        elif self.activation == Activations.Sigmoid:
            return sigmoid_backward(dA, x)
        else:
            raise Exception("Unknown activation function")
        

class Activations(Enum):
    ReLU = "ReLU"
    Sigmoid = "Sigmoid"