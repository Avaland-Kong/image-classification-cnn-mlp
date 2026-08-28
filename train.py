import time


def train_model(model, train_loader, criterion, optimizer, device, epochs):
    # move model to device
    model.to(device)
    # set model to train mode
    model.train()

    # train the model
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # move data to device
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            # calculate loss
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            # update the parameters
            optimizer.step()

            # calculate the total loss, correct predictions and total samples
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {total_loss/total:.4f} | "
            f"Accuracy: {correct/total:.4f} | "
            f"Time: {time.time() - start_time:.2f}s"
        )

    return model
