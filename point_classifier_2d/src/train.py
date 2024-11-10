import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print("Usage: python dump_points.py <file>")
        return

    points, labels = load_data(sys.argv[1])
    model = train_model(points, labels)
    random_points, result = exec_for_random_points(model)

    dump_points(random_points, result)


def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    points = data[:, :2]
    labels = data[:, 2]
    return points, labels


def train_model(points, labels):
    model = PointClassifier2D()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10000):
        optimizer.zero_grad()
        outputs = model(torch.Tensor(points))
        loss = loss_fn(outputs, torch.Tensor(labels))
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


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


class PointClassifier2D(torch.nn.Module):
    def __init__(self):
        super(PointClassifier2D, self).__init__()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x).squeeze()
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    main()
