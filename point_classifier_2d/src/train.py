import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 3:
        print("Usage: python dump_points.py <file> <model>")
        return

    points, labels = load_data(sys.argv[1])
    model = construct_model(sys.argv[2])
    train_model(model, points, labels)
    random_points, result = exec_for_random_points(model)

    dump_points(random_points, result)


def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    points = data[:, :2]
    labels = data[:, 2]
    return points, labels


def construct_model(model_type):
    if model_type == "simple":
        return PointClassifier2DSimple()
    if model_type == "complex":
        return PointClassifier2DComplex()
    raise ValueError(f"Invalid model type: {model_type}")


def train_model(model, points, labels):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10001):
        optimizer.zero_grad()
        outputs = model(torch.Tensor(points))
        loss = loss_fn(outputs, torch.Tensor(labels))
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


def exec_for_random_points(model):
    random_points = np.random.rand(1000, 2)
    result = model(torch.Tensor(random_points)).detach().numpy()
    return random_points, result


def dump_points(points, labels):
    red = points[labels < 0.5]
    blue = points[labels > 0.5]
    plt.scatter(red[:, 0], red[:, 1], c="red")
    plt.scatter(blue[:, 0], blue[:, 1], c="blue")
    plt.show()


class PointClassifier2DSimple(torch.nn.Module):
    def __init__(self):
        super(PointClassifier2DSimple, self).__init__()
        self.fc1 = torch.nn.Linear(2, 1)

    def forward(self, data):
        data = self.fc1(data)
        data = torch.sigmoid(data)
        return data.squeeze()


class PointClassifier2DComplex(torch.nn.Module):
    def __init__(self):
        super(PointClassifier2DComplex, self).__init__()
        self.fc1 = torch.nn.Linear(2, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, data):
        data = self.fc1(data)
        data = torch.relu(data)
        data = self.fc2(data)
        data = torch.relu(data)
        data = self.fc3(data)
        data = torch.sigmoid(data)
        return data.squeeze()


if __name__ == "__main__":
    main()
