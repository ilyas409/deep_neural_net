import numpy as np
from loss import BinaryCrossEntropy

class Model:
    """Sequential binary classifier. Supports any number of `Dense` layers"""
    def __init__(self, layers):
        self.layers = layers
        self.loss_fn = BinaryCrossEntropy()
        inp_size = self.layers[0].input_size
        # Generate weights and biases:
        for i in range(1, len(self.layers)):
            self.layers[i].init_params(inp_size)
            inp_size = self.layers[i].n_units
    
    def compute_cost(self, AL, Y):
        """Compute cost using the loss function"""
        return self.loss_fn.forward(AL, Y)
    
    def evaluate(self, X, Y):
        Y_hat = self.predict(X)
        return round(np.sum(Y_hat == Y) / (Y.shape[1]) * 100, 2)
        
    def predict(self, X):
        Y_hat = self.prop_forward(X)
        return np.where(Y_hat.reshape(1, -1) > 0.5, 1, 0)

    def prop_forward(self, X):
        self.memory = {}
        A = X
        self.layers[0].A = A
        for i in range(1, len(self.layers)):
            A = self.layers[i].compute(A)
        return A
        
    def fit(self, X, Y, learning_rate, iterations):
        self.learning_rate = learning_rate
        costs = []
        for i in range(iterations):
            Y_hat = self.prop_forward(X)
            self.propagate_backward(Y_hat, Y)
            cost = self.compute_cost(Y_hat, Y)
            costs.append(cost)
            if i > 10 and i % int(iterations * 0.1) == 0:
                print("cost after iteration",i,":",cost)
        return costs
    
    def propagate_backward(self, AL, Y):
        """Clean backpropagation with extracted loss computation"""
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        # Use loss function to compute initial gradient
        dA_prev = self.loss_fn.backward(AL, Y)
        
        for i in reversed(range(1, len(self.layers))):
            dA_curr = dA_prev
            
            # Apply dropout gradient scaling if needed
            if (self.layers[i].dropout > 0):
                keep_prob = 1 - self.layers[i].dropout
                dA_curr = dA_curr * self.layers[i].D / keep_prob

            # Get layer parameters
            A_prev = self.layers[i-1].A
            Z_curr = self.layers[i].Z
            W_curr = self.layers[i].W
            
            # Compute gradients
            dZ_curr = self.layers[i].activation.backward(dA_curr, Z_curr)
            dW_curr = 1/m * np.dot(dZ_curr, A_prev.T)
            db_curr = 1/m * np.sum(dZ_curr, axis=1, keepdims=True)
            dA_prev = np.dot(W_curr.T, dZ_curr)
            
            # Update parameters
            self.layers[i].W -= self.learning_rate * dW_curr
            self.layers[i].b -= self.learning_rate * db_curr