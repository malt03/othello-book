import torchvision
import matplotlib.pyplot as plt
import torch
import random

import torchvision.transforms.functional


def main():
    train_data, test_data = load_data()

    model = Mnist()
    train_losses, test_losses, test_accuracies = train_model(
        model, train_data, test_data
    )
    show_results(train_losses, test_losses, test_accuracies)


def target_transform(label):
    match label:
        case 4:
            return torch.tensor(0.0)
        case 9:
            return torch.tensor(1.0)
        case _:
            return torch.tensor(-1.0)


def load_data():
    transform = torchvision.transforms.ToTensor()

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.functional.rgb_to_grayscale,
        ]
    )

    # MNISTデータセットをロード (target_transform を追加)
    train = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    test = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )

    train_indices = [i for i, label in enumerate(train.targets) if label in [4, 9]]
    test_indices = [i for i, label in enumerate(test.targets) if label in [4, 9]]

    train_indices = random.sample(train_indices, 2000)
    test_indices = random.sample(test_indices, 500)

    train = torch.utils.data.Subset(train, train_indices)
    test = torch.utils.data.Subset(test, test_indices)

    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)

    return train_loader, test_loader


def train_model(model, train_data, test_data):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(40):
        sum_loss = 0.0
        model.train()
        for data, label in train_data:
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        train_loss = sum_loss / len(train_data)

        sum_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            model.eval()
            for data, label in test_data:
                pred = model(data)
                loss = loss_fn(pred, label)
                sum_loss += loss.item()
                test_correct += (pred.round() == label).sum().item()
        test_loss = sum_loss / len(test_data)
        test_accuracy = test_correct / len(test_data.dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}"
        )

    return train_losses, test_losses, test_accuracies


def show_results(train_losses, test_losses, test_accuracies):
    _, ax1 = plt.subplots()
    ax1.plot(train_losses, color="blue", label="Train Loss")
    ax1.plot(test_losses, color="green", label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(test_accuracies, color="red", label="Test Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.show()


class Mnist(torch.nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=3)

        self.fc = torch.nn.Linear(16, 1)

    def forward(self, data):
        data = self.conv1(data)
        data = torch.relu(data)
        data = torch.max_pool2d(data, kernel_size=2)
        data = self.conv2(data)
        data = torch.relu(data)
        data = torch.max_pool2d(data, kernel_size=2)
        data = self.conv3(data)
        data = torch.relu(data)
        data = torch.max_pool2d(data, kernel_size=2)
        data = data.view(-1, 16)
        data = self.fc(data)
        data = torch.sigmoid(data)
        return data.squeeze()


if __name__ == "__main__":
    main()
