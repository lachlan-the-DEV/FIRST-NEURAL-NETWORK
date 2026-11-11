"""       INPUT       HIDDEN LAYER       OUTPUT

         x  ──────┬── Neuron 1 ──┐
                  │              │
                  └── Neuron 2 ──┴── Prediction


This network has 1 input, 2 hidden neurons on 1 layer and 1 output, built 11/11/26
"""
import numpy as np

#training data

#this is a matrix, has the dimensions 5x1
inputs = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

targets = np.array([
    [3],
    [5],
    [7],
    [9],
    [11]
])


# initilise weights and biases
W1 = np.random.randn(1, 2) #### this is the first weight matrix, is (1, 2) dimensions because of 1 input and 2 hidden neurons
b1 = np.zeros((1, 2)) #### one bias for each hidden neuron

W2 = np.random.randn(2, 1) #### two hidden neurons connecting to one output neurons
b2 = np.zeros((1, 1)) #### one output neuron so one bias

# FORWARD PROPAGATION

# Hidden Layer
z1 = inputs @ W1 + b1 ####### This is the same as z1 = WX + b
a1 = np.maximum(0, z1) ###### This does ReLU ro every element of the matrix

# Output Layer
prediction = a1 @ W2 + b2

#Calculate loss
loss = np.mean((prediction - targets) **2)

# Backpropogation

# Gradient of loss with respect to predictions
d_prediction = (2 / len(inputs)) * (prediction - targets) # same as 2/N (prediction - target) with N being amount of number of training examples

# gradient for the output layer
dW2 = a1.T @ d_prediction

db2 = np.sum(d_prediction, axis=0, keepdims=True)

# Gradients for hidden layer
d_a1 = d_prediction @ W2.T # how much the loss chanfes when each hidden neurons output changes
relu_derivative = (z1 > 0).astype(float) #creates a matrix where 1 wherever the OG valur was posotive 0 when 0 or negative
d_z1 = d_a1 * relu_derivative #chain rule

dW1 = inputs.T @ d_z1 #this and below calculate hidden layers weights and biases
db1 = np.sum(d_z1, axis=0, keepdims=True)

learning_rate = 0.01

W1 -= learning_rate * dW1
b1 -= learning_rate * db1

W2 -= learning_rate * dW2
b2 -= learning_rate * db2

for epoch in range(1000):

    # FORWARD PROPAGATION
    z1 = inputs @ W1 + b1
    a1 = np.maximum(0, z1)

    prediction = a1 @ W2 + b2

    # CALCULATE LOSS
    loss = np.mean((prediction - targets) ** 2)

    # BACKPROPAGATION
    d_prediction = (2 / len(inputs)) * (prediction - targets)

    dW2 = a1.T @ d_prediction
    db2 = np.sum(d_prediction, axis=0, keepdims=True)

    d_a1 = d_prediction @ W2.T
    relu_derivative = (z1 > 0).astype(float)
    d_z1 = d_a1 * relu_derivative

    dW1 = inputs.T @ d_z1
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # UPDATE WEIGHTS
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
print("\nFinal predictions:")
print(prediction)

print("\nActual targets:")
print(targets)
