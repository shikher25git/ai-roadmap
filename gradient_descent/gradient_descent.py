"""
Evolution of changes to improve MNIST accuracy from ~30% to ~90%:

1. Initial Version (First Draft):
   - Basic weight initialization using np.random.rand(10, layers[0]) → uniform [0,1] distribution
   - Random bias initialization using np.random.rand(10, 1) → uniform [0,1] distribution
   - Learning rate 0.12 → chosen arbitrarily
   - 50 epochs → minimal training time for testing
   - Full batch gradient descent → simplest implementation
   - No input normalization → raw pixel values [0-255]
   - Basic MSE gradient computation → standard choice for regression
   - Problem: Accuracy ~10% due to poor initialization and training dynamics

2. First Improvement Attempt:
   - Changed weight init to np.random.randn * 0.01 → normal distribution for better spread
   - Changed bias init to zeros → prevent initial bias in predictions
   - Reduced learning rate to 0.01 → prevent overshooting due to large gradients
   - Increased epochs to 200 → give more time to converge
   - Problem: Still low accuracy because inputs weren't normalized and gradients were unstable

3. Second Attempt:
   - Increased weight init scale to 0.1 → weights too small in previous attempt
   - Changed bias init to small random values (0.1) → break symmetry in neuron outputs
   - Increased learning rate to 0.05 → previous rate too conservative
   - Added scaling factor of 2 to gradient computation → attempt to prevent vanishing gradients
   - Increased epochs to 500 → allow more time to converge
   - Problem: Slight improvement but still suboptimal due to lack of normalization

4. Final Version (Current):
   - Input normalization (X / 255.0) → crucial for stable gradients with sigmoid
   - Xavier/Glorot initialization for weights → scale based on input size (np.sqrt(1.0/layers[0]))
   - Zero initialization for biases → found random biases unnecessary
   - Mini-batch gradient descent (batch_size=32) → better gradient estimates than full batch
   - Data shuffling per epoch → prevent learning order bias
   - Learning rate 0.1 → appropriate for normalized inputs
   - 500 epochs → sufficient for convergence
   - Proper gradient computation → removed arbitrary scaling
   - Result: Accuracy ~90% due to stable training dynamics

Key improvements that led to success:
1. Input normalization - crucial for sigmoid activation (prevents saturation)
2. Mini-batch processing - better convergence and training stability (noise helps escape local minima)
3. Proper weight initialization - prevents vanishing gradients (maintains variance across layers)
4. Data shuffling - prevents learning order bias (better generalization)
5. Corrected gradient computation - proper backpropagation (mathematically sound updates)

Main takeaways:
1. Normalization is crucial for neural networks → keeps inputs in good range for activation functions
2. Proper initialization helps with training dynamics → prevents vanishing/exploding gradients
3. Mini-batches provide better gradient estimates → balance between computation and accuracy
4. Even a single-layer network can achieve good accuracy → proper optimization more important than depth
5. Each hyperparameter change should have clear mathematical/statistical justification
"""

import numpy as np

class Network:
    def __init__(self, layers: list[int]):
        """
        Layers is a list of integers which denotes the number of layers and the number of neurons in each layer.
        """
        self.layers = layers

        # Initialize weights with proper scaling
        self.weights = np.random.randn(10, layers[0]) * np.sqrt(1.0/layers[0])
        # Initialize biases to zero
        self.biases = np.zeros((10, 1))
        
        self.epochs = 500
        # Reduce learning rate for normalized inputs
        self.lr = 0.1
        self.batch_size = 32
        
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))


    # def cost(self, predicted, actual, input):
    #     delta = 2 * (predicted - actual.T) * predicted * (1 - predicted)
        
    #     weight_diff = np.dot(delta, input.T)  # (10, 784)
    #     bias_diff = np.sum(delta, axis=1, keepdims=True)  # (10, 1)
    
    #     return (
    #         np.mean(weight_diff),
    #         np.mean(bias_diff)
    #     )

    def cost(self, predicted, actual, input):
        m = input.shape[1]  # batch size
        
        # Compute gradients
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
            
            # Process in batches
            for j in range(n_batches):
                start_idx = j * self.batch_size
                end_idx = start_idx + self.batch_size
                
                X_batch = X_shuffled[start_idx:end_idx].T
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                predicted = self.predict(X_batch)
                weight, bias = self.cost(predicted, Y_batch, X_batch)
                self.weights -= weight * self.lr
                self.biases -= bias * self.lr
            
            # Print progress every 10 epochs
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


        