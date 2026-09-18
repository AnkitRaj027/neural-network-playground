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
import matplotlib.pyplot as plt


def create_training_plots(train_losses, test_accuracies):

    # Create loss figure
    loss_fig, loss_ax = plt.subplots(figsize=(7, 4))

    loss_ax.plot(train_losses)

    loss_ax.set_title("Training Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")

    loss_ax.grid(True)


    # Create accuracy figure
    accuracy_fig, accuracy_ax = plt.subplots(figsize=(7, 4))

    accuracy_ax.plot(test_accuracies)

    accuracy_ax.set_title("Test Accuracy")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Accuracy")

    accuracy_ax.set_ylim(0, 1)

    accuracy_ax.grid(True)


    return loss_fig, accuracy_fig

def plot_decision_boundary(model, X, y):

    # Get the minimum and maximum values
    x_min = X[:, 0].min().item() - 0.5
    x_max = X[:, 0].max().item() + 0.5

    y_min = X[:, 1].min().item() - 0.5
    y_max = X[:, 1].max().item() + 0.5


    # Create a grid of points
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )


    # Convert grid into input format
    grid = np.c_[
        xx.ravel(),
        yy.ravel()
    ]

    grid = torch.tensor(
        grid,
        dtype=torch.float32
    )


    # Make predictions
    model.eval()

    with torch.no_grad():

        predictions = model(grid)

        predicted_classes = torch.argmax(
            predictions,
            dim=1
        )


    # Convert predictions back into grid shape
    predicted_classes = predicted_classes.numpy().reshape(
        xx.shape
    )


    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))


    # Draw decision regions
    ax.contourf(
        xx,
        yy,
        predicted_classes,
        alpha=0.3
    )


    # Draw actual data points
    ax.scatter(
        X[:, 0].numpy(),
        X[:, 1].numpy(),
        c=y.numpy(),
        edgecolors="black"
    )


    ax.set_title("Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


    return fig