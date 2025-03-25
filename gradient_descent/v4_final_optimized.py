"""
Final optimized version with proper initialization, normalization, and mini-batch processing.
Characteristics:
- Input normalization (X / 255.0)
- Xavier/Glorot initialization for weights
- Zero initialization for biases
- Mini-batch gradient descent
- Data shuffling
- Proper learning rate
- Mathematically sound gradient computation
- Achieved ~90% accuracy
"""

import numpy as np

class Network:
    def __init__(self, layers: list[int]):
        """
        Layers is a list of integers which denotes the number of layers and the number of neurons in each layer.
        """
        self.layers = layers
        # Xavier/Glorot initialization
        self.weights = np.random.randn(10, layers[0]) * np.sqrt(1.0/layers[0])
        # Zero initialization for biases
        self.biases = np.zeros((10, 1))
        
        self.epochs = 500
        self.lr = 0.1
        self.batch_size = 32
        
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def cost(self, predicted, actual, input):
        m = input.shape[1]  # batch size
        
        # Proper gradient computation
        dz = (predicted - actual.T) * predicted * (1 - predicted)  # shape: (10, m)
        dw = (1/m) * np.dot(dz, input.T)  # shape: (10, 784)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)  # shape: (10, 1)
        
        return dw, db

    def predict(self, X):
        value = np.dot(self.weights, X) + self.biases
        return self.sigmoid(value)
        
    def train(self, X, Y):
        # Normalize input data
        X = X / 255.0
        
        n_samples = X.shape[0]
        n_batches = n_samples // self.batch_size
        
        for i in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            # Process in mini-batches
            for j in range(n_batches):
                start_idx = j * self.batch_size
                end_idx = start_idx + self.batch_size
                
                X_batch = X_shuffled[start_idx:end_idx].T
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                predicted = self.predict(X_batch)
                weight, bias = self.cost(predicted, Y_batch, X_batch)
                self.weights -= weight * self.lr
                self.biases -= bias * self.lr
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}/{self.epochs}")

    def test(self, X, Y):
        # Normalize test data
        X = X / 255.0
        
        predicted = self.predict(X.T)
        predicted_labels = np.argmax(predicted, axis=0)
        actual_labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predicted_labels == actual_labels) * 100
        print(f"Accuracy: {accuracy:.2f}%") 