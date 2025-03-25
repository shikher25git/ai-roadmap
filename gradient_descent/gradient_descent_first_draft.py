import numpy as np

class Network:
    def __init__(self, layers: list[int]):
        """
        Layers is a list of integers which denotes the number of layers and the number of neurons in each layer.
        """
        self.layers = layers

        self.weights = np.random.rand(10, layers[0])  # Random weight initialization
        self.biases = np.random.rand(10, 1)  # Random bias
        
        self.epochs = 50
        self.lr = 0.12
        
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def cost(self, predicted, actual, input):
        delta = 2 * (predicted - actual.T) * predicted * (1 - predicted)
        
        weight_diff = np.dot(delta, input.T)  # (10, 784)
        bias_diff = np.sum(delta, axis=1, keepdims=True)  # (10, 1)
    
        return (
            np.mean(weight_diff),
            np.mean(bias_diff)
        )

    def predict(self, X):
        value = np.dot(self.weights, X) + self.biases
        return self.sigmoid(value)
        
    def train(self, X, Y):
        for i in range(self.epochs):
            predicted = self.predict(X.T)
            weight, bias = self.cost(predicted, Y, X.T)
            self.weights -= weight * self.lr  # Subtract gradient
            self.biases -= bias * self.lr  # Subtract gradient

    def test(self, X, Y):
        predicted = self.predict(X.T)  # Get model predictions
    
        # Convert predictions to class labels
        predicted_labels = np.argmax(predicted, axis=0)  # Shape (10000,)
        actual_labels = np.argmax(Y, axis=1)  # Ensure correct shape (10000,)
    
        # Calculate accuracy
        accuracy = np.mean(predicted_labels == actual_labels) * 100
    
        print(f"Accuracy: {accuracy:.2f}%") 