import numpy as np

class Network:
    def __init__(self, layers: list[int]):
        """
        Layers is a list of integers which denotes the number of layers and the number of neurons in each layer.
        """
        self.layers = layers

        self.weights = np.random.rand(10, layers[0])  # Random weight initialization
        self.biases = np.random.rand(10, 1)  # Random bias
        
        # self.biases = [
        #     np.random.rand(neuron_count, 1)
        #     for neuron_count in layers[1:]
        # ]
        
        # self.weights = [
        #     np.random.rand(layers[i-1], layers[i]) for i in range(1, len(self.layers))
        # ]
        self.epochs = 1000
        self.lr = 0.5
        
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def weight_gradient(self, X):
        # print(X.shape)
        sig = np.exp(
            -(np.dot(self.weights, X) + self.biases)
        )
        denom = (1 + sig) ** (2)
        # print(denom.shape)
        numerator = np.dot(sig/denom, X.T)
        return numerator

    def bias_gradient(self, X):
        sig = np.exp(
            -(np.dot(self.weights, X) + self.biases)
        )
        denom = (1 + sig) ** (2)
        numerator = sig
        delta = numerator / denom
        return delta.T

    def cost(self, predicted, actual, input):
        # print(actual.shape)
        # print((1-predicted.shape)
        # print(input.shape)
        # print(self.weights.shape), (1-predicted)) * 2
        bias_diff = np.dot((actual.T - predicted), predicted.T) * 2
        print(bias_diff.shape)
        weight_diff = np.dot((actual.T - predicted), predicted.T, (1-predicted), input) * 2

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
            self.weights += weight * self.lr
            self.biases += bias * self.lr
            # print(f"Epoch {i} done")


    def test(self, X, Y):
        predicted = self.predict(X.T)  # Get model predictions
    
        # Convert predictions to class labels
        predicted_labels = np.argmax(predicted, axis=0)  # Shape (10000,)
        actual_labels = np.argmax(Y, axis=1)  # Ensure correct shape (10000,)
    
        # Calculate accuracy
        accuracy = np.mean(predicted_labels == actual_labels) * 100
    
        print(f"Accuracy: {accuracy:.2f}%")

        