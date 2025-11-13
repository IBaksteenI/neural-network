import torch
from torch import nn, optim
from torchvision import datasets, transforms
from models.neural_network import NeuralNet

def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # epoch means one complete pass through the training dataset
    # Here we train for 25 epochs
    epochs = 25
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            
            # Compute the loss
            # what is loss and what does it do?
            # Loss is a measure of how well the model's predictions match the true labels.
            # It is used to train the model by adjusting its parameters to minimize the loss.
            # Here we use CrossEntropyLoss which is suitable for multi-class classification problems.
            # It combines LogSoftmax and NLLLoss in one single class.
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")

    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved as mnist_model.pth")

if __name__ == "__main__":
    train_model()