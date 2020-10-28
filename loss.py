import numpy as np

class BinaryCrossEntropy:
    """Binary cross-entropy loss function with numerical stability"""
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        """Compute binary cross-entropy loss"""
        m = y_pred.shape[1]
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        cost = -1/m * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped), axis=0)
        return np.squeeze(cost)
    
    def backward(self, y_pred, y_true):
        """Compute gradient of binary cross-entropy loss"""
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        pos_term = y_true / y_pred_clipped
        neg_term = (1 - y_true) / (1 - y_pred_clipped)
        return -(pos_term - neg_term)