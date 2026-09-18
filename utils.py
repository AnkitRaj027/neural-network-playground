import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_decision_boundary(model, X, y):

    # Find the boundaries of our dataset
    x_min = X[:, 0].min().item() - 0.5
    x_max = X[:, 0].max().item() + 0.5

    y_min = X[:, 1].min().item() - 0.5
    y_max = X[:, 1].max().item() + 0.5


    # Create a grid of points
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )


    # Convert grid into PyTorch tensor
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    )


    # Make predictions
    with torch.no_grad():

        predictions = model(grid)

        classes = torch.argmax(
            predictions,
            dim=1
        )


    # Reshape predictions
    Z = classes.numpy().reshape(
        xx.shape
    )


    # Draw decision regions
    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        cmap="coolwarm"
    )


    # Draw actual data points
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="black"
    )


    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Neural Network Decision Boundary")

    plt.show()