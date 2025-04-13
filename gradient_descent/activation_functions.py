"""
Implementation and visualization of common activation functions and their gradients.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    """Gradient of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_gradient(x):
    """Gradient of ReLU function."""
    # Convert to numpy array if scalar
    x = np.array(x)
    # Handle both scalar and array inputs
    return np.where(x > 0, 1.0, 0.0)

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def tanh_gradient(x):
    """Gradient of tanh function."""
    return 1 - np.tanh(x) ** 2

def plot_activation_functions():
    """Plot activation functions and their gradients."""
    # Create two sets of points to better show the discontinuity
    x_neg = np.linspace(-5, -0.01, 50)
    x_pos = np.linspace(0.01, 5, 50)
    x = np.concatenate([x_neg, x_pos])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot activation functions
    ax1.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
    ax1.plot(x, relu(x), label='ReLU', linewidth=2)
    ax1.plot(x, tanh(x), label='Tanh', linewidth=2)
    
    ax1.set_title('Activation Functions')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot gradients
    ax2.plot(x, sigmoid_gradient(x), label='Sigmoid Gradient', linewidth=2)
    # Plot ReLU gradient separately for negative and positive regions
    ax2.plot(x_neg, relu_gradient(x_neg), 'b-', label='ReLU Gradient', linewidth=2)
    ax2.plot(x_pos, relu_gradient(x_pos), 'b-', linewidth=2)
    ax2.plot(x, tanh_gradient(x), label='Tanh Gradient', linewidth=2)
    
    # Add a point at x=0 to show the discontinuity
    ax2.plot(0, 0, 'bo', markersize=8)
    ax2.plot(0, 1, 'bo', markersize=8, fillstyle='none')
    
    ax2.set_title('Gradients of Activation Functions')
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def check_gradients_at_points(points=[-2, -1, 0, 1, 2]):
    """Print activation values and gradients at specific points."""
    print("\nActivation Values and Gradients at Different Points:")
    print("Point\tSigmoid\t\tSigmoid Grad\tReLU\t\tReLU Grad\tTanh\t\tTanh Grad")
    print("-" * 100)
    
    for x in points:
        sig_val = sigmoid(x)
        sig_grad = sigmoid_gradient(x)
        relu_val = relu(x)
        relu_grad = relu_gradient(x)
        tanh_val = tanh(x)
        tanh_grad = tanh_gradient(x)
        
        print(f"{x:5.1f}\t{sig_val:8.4f}\t{sig_grad:8.4f}\t{relu_val:8.4f}\t{relu_grad:8.4f}\t{tanh_val:8.4f}\t{tanh_grad:8.4f}")

if __name__ == "__main__":
    # Plot activation functions and their gradients
    fig = plot_activation_functions()
    plt.show()
    
    # Check values at specific points
    check_gradients_at_points()
    
    # Additional points to check
    check_gradients_at_points([-10, -5, 5, 10]) 