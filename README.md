# deep_neural_net
Multilayer neural network


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
