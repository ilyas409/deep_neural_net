# deeplearn-scratch

A toy implementation of a multilayer neural network library, built for learning purposes and inspired by Keras. I was motivated to build this after taking Andrew Ng's Deep Learning Specialization on Coursera (https://www.coursera.org/specializations/deep-learning). While the course is taught in TensorFlow 1, it also introduces students to building neural networks from scratch in NumPy, which inspired me to attempt my own toy NumPy implementation with a Keras-like API.
This project implements a simple sequential model with dense layers, various activation functions, dropout regularization, and backpropagation from scratch using NumPy.
## Features

- **Sequential Model Architecture**: Build networks layer by layer
- **Dense (Fully Connected) Layers**: Standard neural network building blocks
- **Activation Functions**: ReLU and Sigmoid with proper gradient computation
- **Dropout Regularization**: Prevent overfitting during training
- **Binary Classification**: Optimized for binary cross-entropy loss
- **Numerical Stability**: Proper clipping and epsilon handling

This is a educational implementation designed to understand the fundamentals of neural networks without relying on high-level frameworks.

## Example Usage

### Import the modules
```python3
from deep_neural_net.activations import *
from deep_neural_net.model import *
from deep_neural_net.layer import *
from deep_neural_net import lr_utils
```

### Load a dataset
```python3
train_x_orig, train_y, test_x_orig, test_y, classes = lr_utils.load_dataset()
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
```

### Define a model
```python3
model = Model([
    Input(len(train_x)),
    Dense(64, activation=Activation(Activations.ReLU), weight_multiplier=0.1, dropout=0.15),
    Dense(16, activation=Activation(Activations.ReLU), weight_multiplier=0.1, dropout=0.2),
    Dense(5, activation=Activation(Activations.ReLU), weight_multiplier=0.1),
    Dense(1, activation=Activation(Activations.Sigmoid), weight_multiplier=1)
])
```

### Train the model
```python3
c = model.fit(train_x, train_y, learning_rate=0.0075, iterations=2500)
```

### Evaluate the model
```python3
print(model.evaluate(train_x, train_y), model.evaluate(test_x, test_y))
```

## Learning Goals

This implementation helped me understand:
- Forward and backward propagation mechanics
- Gradient computation and chain rule application
- Weight initialization strategies
- Regularization techniques (dropout)
- Numerical stability considerations in deep learning

*Inspired by the Keras API design but implemented from scratch for educational purposes.*