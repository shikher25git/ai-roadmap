"""
Script to visualize gradient descent convergence and the effect of different learning rates.

Note on High Learning Rate Stability:
The implementation shows stable convergence even with high learning rates (e.g., 1.0) due to three key factors:

1. Input Normalization (X / 255.0):
   - Normalizes input to [0,1] range
   - Ensures well-scaled gradients
   - Prevents gradient explosion from raw pixel values (0-255)

2. Mini-batch Processing (batch_size=32):
   - Provides stable gradient estimates
   - Adds controlled noise to help escape local minima
   - More stable than full-batch or single-sample updates

3. Single Layer Architecture:
   - No vanishing/exploding gradient problems
   - Direct gradient flow from output to weights/biases
   - No chain rule multiplication through multiple layers

Without these safeguards (e.g., using raw pixel values, full-batch updates, or a deeper network),
high learning rates would likely cause oscillation or divergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import time

# Network class definition
class Network:
    def __init__(self, layers: list[int]):
        """
        Layers is a list of integers which denotes the number of neurons in each layer.
        For MNIST: layers[0] = 784 (input size)
        """
        self.layers = layers
        # Xavier/Glorot initialization
        self.weights = np.random.randn(10, layers[0]) * np.sqrt(1.0/layers[0])
        # Zero initialization for biases
        self.biases = np.zeros((10, 1))
        
        self.epochs = 100  # Default epochs
        self.lr = 0.1      # Default learning rate
        self.batch_size = 32
        
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def cost(self, predicted, actual, input):
        m = input.shape[1]  # batch size
        # Proper gradient computation
        dz = (predicted - actual.T) * predicted * (1 - predicted)
        dw = (1/m) * np.dot(dz, input.T)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        return dw, db

    def predict(self, X):
        value = np.dot(self.weights, X) + self.biases
        return self.sigmoid(value)

def load_mnist_subset(n_samples=5000):
    """Load a subset of MNIST data."""
    print("Loading MNIST data...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Take a subset of data
    indices = np.random.permutation(len(X))[:n_samples]
    X = X[indices]
    y = y[indices]
    
    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y.reshape(-1, 1))
    
    return X, Y

def train_and_track_loss(network, X, Y, learning_rate):
    """Train network and track loss over iterations."""
    network.lr = learning_rate
    losses = []
    accuracies = []  # Track accuracies too
    
    # Normalize input
    X = X / 255.0
    n_samples = X.shape[0]
    n_batches = n_samples // network.batch_size
    
    for epoch in range(network.epochs):
        epoch_loss = 0
        correct_predictions = 0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        for j in range(n_batches):
            start_idx = j * network.batch_size
            end_idx = start_idx + network.batch_size
            
            X_batch = X_shuffled[start_idx:end_idx].T
            Y_batch = Y_shuffled[start_idx:end_idx]
            
            # Forward pass
            predicted = network.predict(X_batch)
            
            # Compute MSE loss
            loss = np.mean((predicted - Y_batch.T) ** 2)
            epoch_loss += loss
            
            # Compute accuracy
            predicted_labels = np.argmax(predicted, axis=0)
            actual_labels = np.argmax(Y_batch, axis=1)
            correct_predictions += np.sum(predicted_labels == actual_labels)
            
            # Backward pass
            weight, bias = network.cost(predicted, Y_batch, X_batch)
            network.weights -= weight * network.lr
            network.biases -= bias * network.lr
        
        avg_epoch_loss = epoch_loss / n_batches
        accuracy = correct_predictions / (n_batches * network.batch_size) * 100
        
        losses.append(avg_epoch_loss)
        accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return losses, accuracies

def plot_training_curves(learning_rates=[0.001, 0.01, 0.1, 1.0], n_samples=5000):
    """Plot both loss and accuracy curves for different learning rates."""
    X, Y = load_mnist_subset(n_samples)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        network = Network([784])  # MNIST images are 28x28 = 784 pixels
        losses, accuracies = train_and_track_loss(network, X, Y, lr)
        
        # Plot loss
        ax1.plot(losses, label=f'lr = {lr}')
        # Plot accuracy
        ax2.plot(accuracies, label=f'lr = {lr}')
    
    # Loss plot
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Loss Convergence')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Accuracy plot
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    fig = plot_training_curves(learning_rates, n_samples=5000)
    plt.show() 