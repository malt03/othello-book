import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print("Usage: python dump_points.py <file>")
        return

    file = sys.argv[1]
    data = np.loadtxt(file, delimiter=",")

    points = data[:, :2]
    labels = data[:, 2]

    red = points[labels == 0]
    blue = points[labels == 1]
    plt.scatter(red[:, 0], red[:, 1], c="red")
    plt.scatter(blue[:, 0], blue[:, 1], c="blue")
    plt.show()


if __name__ == "__main__":
    main()
