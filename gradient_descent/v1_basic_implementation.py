"""
Initial implementation with basic uniform initialization and full batch gradient descent.
Characteristics:
- Uniform [0,1] weight and bias initialization
- No input normalization
- Full batch gradient descent
- Basic MSE gradient computation
- Accuracy: ~10%
"""

import numpy as np

class Network:
    def __init__(self, layers: list[int]):
        """
        Layers is a list of integers which denotes the number of layers and the number of neurons in each layer.
        """
        self.layers = layers
        # Uniform initialization [0,1]
        self.weights = np.random.rand(10, layers[0])
        self.biases = np.random.rand(10, 1)
        
        self.epochs = 50
        self.lr = 0.12
        
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def cost(self, predicted, actual, input):
        # Basic MSE gradient computation
        delta = 2 * (predicted - actual.T) * predicted * (1 - predicted)
        weight_diff = np.dot(delta, input.T)
        bias_diff = np.sum(delta, axis=1, keepdims=True)
    
        return (
            np.mean(weight_diff),
            np.mean(bias_diff)
        )

    def predict(self, X):
        value = np.dot(self.weights, X) + self.biases
        return self.sigmoid(value)
        
    def train(self, X, Y):
        # No input normalization
        for i in range(self.epochs):
            # Full batch gradient descent
            predicted = self.predict(X.T)
            weight, bias = self.cost(predicted, Y, X.T)
            self.weights -= weight * self.lr
            self.biases -= bias * self.lr
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}/{self.epochs}")

    def test(self, X, Y):
        predicted = self.predict(X.T)
        predicted_labels = np.argmax(predicted, axis=0)
        actual_labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predicted_labels == actual_labels) * 100
        print(f"Accuracy: {accuracy:.2f}%") 