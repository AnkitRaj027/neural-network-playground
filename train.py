import torch


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    learning_rate,
    epochs
):

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    train_losses = []
    test_accuracies = []


    for epoch in range(epochs):

        # Training mode
        model.train()

        # Forward pass
        predictions = model(X_train)

        # Calculate loss
        loss = loss_fn(
            predictions,
            y_train
        )

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()


        # Evaluate model
        model.eval()

        with torch.no_grad():

            test_predictions = model(X_test)

            test_classes = torch.argmax(
                test_predictions,
                dim=1
            )

            accuracy = (
                test_classes == y_test
            ).float().mean()


        # Store metrics
        train_losses.append(
            loss.item()
        )

        test_accuracies.append(
            accuracy.item()
        )


    return (
        model,
        train_losses,
        test_accuracies
    )