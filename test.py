import torch

@torch.no_grad()
def test_model(model, test_loader, criterion, device):
    # move model to device
    model.to(device)
    # set model to evaluation mode
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        # move data to device
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)
        # calculate loss
        loss = criterion(outputs, labels)

        # calculate the total loss, correct predictions and total samples
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    print(
        f"Test Loss: {avg_loss:.4f} | "
        f"Test Accuracy: {accuracy:.4f}"
    )

    return avg_loss, accuracy
