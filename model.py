import numpy as np

class Model:
    """Sequential binary classifier. Supports any number of `Dense` layers"""
    def __init__(self, layers, use_adam_optimizer=True):
        self.layers = layers
        self.use_adam_optimizer = use_adam_optimizer
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
    
    def update_parameters(self, i, dW, db):
        if not self.use_adam_optimizer:
            self.layers[i].W -= self.learning_rate * dW
            self.layers[i].b -= self.learning_rate * db
            return
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
    
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        self.layers[i].vdW = beta1 * self.layers[i].vdW + (1 - beta1) * dW
        self.layers[i].vdb = beta1 * self.layers[i].vdb + (1 - beta1) * db

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected_dW = self.layers[i].vdW / (1 - beta1)
        v_corrected_db = self.layers[i].vdb / (1 - beta1)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        self.layers[i].sdW = beta2 * self.layers[i].sdW + (1 - beta2) * np.power(dW, 2)
        self.layers[i].sdb = beta2 * self.layers[i].sdb + (1 - beta2) * np.power(db, 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected_dW = self.layers[i].sdW / (1 - beta2)
        s_corrected_db = self.layers[i].sdb / (1 - beta2)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        self.layers[i].W = self.layers[i].W - self.learning_rate * v_corrected_dW / (np.sqrt(s_corrected_dW) + epsilon)
        self.layers[i].b = self.layers[i].b - self.learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + epsilon)
    
    def propagate_backward(self, AL, Y):
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        epsilon = 1e-8
        dA_prev = - (np.divide(Y, AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon))
        
        for i in reversed(range(1, len(self.layers))):
            dA_curr = dA_prev
            if (self.layers[i].dropout > 0):
                keep_prob = 1 - self.layers[i].dropout
                dA_curr = dA_curr * self.layers[i].D / keep_prob

            A_prev = self.layers[i-1].A
            Z_curr = self.layers[i].Z
            W_curr = self.layers[i].W
            b_curr = self.layers[i].b
            
            dZ_curr = self.layers[i].activation.backward(dA_curr, Z_curr)
            dW_curr = 1/m * np.dot(dZ_curr, A_prev.T)
            db_curr = 1/m * np.sum(dZ_curr, axis=1, keepdims=True)
            dA_prev = np.dot(W_curr.T, dZ_curr)
            
            self.update_parameters(i, dW_curr, db_curr)