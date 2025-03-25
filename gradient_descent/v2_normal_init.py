"""
Second iteration with normal distribution initialization and reduced learning rate.
Characteristics:
- Normal distribution weight initialization (scaled by 0.01)
- Zero-initialized biases
- Reduced learning rate to prevent overshooting
- Increased epochs for better convergence
- Still low accuracy due to lack of normalization
"""

import numpy as np

class Network:
    def __init__(self, layers: list[int]):
        """
        Layers is a list of integers which denotes the number of layers and the number of neurons in each layer.
        """
        self.layers = layers
        # Normal distribution initialization with small scale
        self.weights = np.random.randn(10, layers[0]) * 0.01
        # Zero initialization for biases
        self.biases = np.zeros((10, 1))
        
        self.epochs = 200  # Increased epochs
        self.lr = 0.01     # Reduced learning rate
        
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def cost(self, predicted, actual, input):
        delta = (predicted - actual.T) * predicted * (1 - predicted)
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
        # Still no input normalization
        for i in range(self.epochs):
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