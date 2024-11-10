import numpy as np
import matplotlib.pyplot as plt


def main():
    points = np.random.rand(200, 2)
    labels = np.random.randint(0, 2, 200).astype(np.float32)
    save_csv(points, labels, "data.csv")

    # # プロット
    red = points[labels == 0]
    blue = points[labels == 1]
    plt.scatter(red[:, 0], red[:, 1], c="red", label="Class 0")
    plt.scatter(blue[:, 0], blue[:, 1], c="blue", label="Class 1")
    plt.show()


def save_csv(points, labels, filename):
    np.savetxt(filename, np.hstack([points, labels.reshape(-1, 1)]), delimiter=",")


if __name__ == "__main__":
    main()
