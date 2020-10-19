import numpy as np

class Model:
    """Sequential binary classifier. Supports any number of `Dense` layers"""
    def __init__(self, layers):
        self.layers = layers
        inp_size = self.layers[0].input_size
        # Generate weights and biases:
        for i in range(1, len(self.layers)):
            self.layers[i].init_params(inp_size)
            inp_size = self.layers[i].n_units
    
    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        cost = - 1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis=1)
        return np.squeeze(cost)
    
    def evaluate(self, X, Y):
        Y_hat = self.predict(X)
        return round(np.sum(Y_hat == Y) / (Y.shape[1]) * 100, 2)
        
    def predict(self, X):
        Y_hat = self.prop_forward(X)
        return np.where(Y_hat.reshape(1, -1) > 0.5, 1, 0)

    def prop_forward(self, X):
        self.memory = {}
        A_curr = X
        for i in range(1, len(self.layers)):
            A_prev = A_curr
            A_curr = self.layers[i].compute(A_prev)
            Z_curr = self.layers[i].Z
            self.memory["A" + str(i-1)] = A_prev
            self.memory["Z" + str(i)] = Z_curr
            
        return A_curr
    
    def update_params(self, grads, learning_rate):
        for i in range(1, len(self.layers)):
            self.layers[i].W -= learning_rate * grads["dW" + str(i)]
            self.layers[i].b -= learning_rate * grads["db" + str(i)]
    
    def fit(self, X, Y, learning_rate, iterations):
        self.learning_rate = learning_rate
        grads = {}
        costs = []
        for i in range(iterations):
            Y_hat = self.prop_forward(X)
            grads = self.propagate_backward(Y_hat, Y)
            self.update_params(grads, learning_rate)
            cost = self.compute_cost(Y_hat, Y)
            costs.append(cost)
            if i > 10 and i % int(iterations * 0.1) == 0:
                print("cost after iteration",i,":",cost)
        return costs
    
    def propagate_backward(self, AL, Y):
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        grads_values = {}
        
        dA_prev = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        for i in reversed(range(1, len(self.layers))):
            
            dA_curr = dA_prev
            
            A_prev = self.memory["A" + str(i-1)]
            Z_curr = self.memory["Z" + str(i)]
            W_curr = self.layers[i].W
            b_curr = self.layers[i].b
            
            dZ_curr = self.layers[i].activation.backward(dA_curr, Z_curr)
            dW_curr = 1/m * np.dot(dZ_curr, A_prev.T)
            db_curr = 1/m * np.sum(dZ_curr, axis=1, keepdims=True)
            dA_prev = np.dot(W_curr.T, dZ_curr)
            
            grads_values["dW" + str(i)] = dW_curr
            grads_values["db" + str(i)] = db_curr
            
        return grads_values