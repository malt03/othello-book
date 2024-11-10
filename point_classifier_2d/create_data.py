import numpy as np
import matplotlib.pyplot as plt


def main():
    points = np.random.rand(200, 2)
    labels = np.zeros(200)

    # ラベル付けをするループ
    for i in range(200):
        # 基本的な境界を設定
        boundary = points[i, 0] * 0.8 + 0.1
        # 距離に応じてラベルを決定（境界近くで確率的にラベルを反転）
        if points[i, 1] > boundary:
            labels[i] = 1
            if np.abs(points[i, 1] - boundary) < 0.1:  # 境界に近い場合
                labels[i] = np.random.choice([0, 1], p=[0.3, 0.7])  # 30%の確率で0に
        else:
            if np.abs(points[i, 1] - boundary) < 0.1:  # 境界に近い場合
                labels[i] = np.random.choice([0, 1], p=[0.7, 0.3])  # 30%の確率で1に

    save_csv(points, labels, "data/linear.csv")

    red = points[labels == 0]
    blue = points[labels == 1]
    plt.scatter(red[:, 0], red[:, 1], c="red")
    plt.scatter(blue[:, 0], blue[:, 1], c="blue")
    plt.show()


def save_csv(points, labels, filename):
    np.savetxt(filename, np.hstack([points, labels.reshape(-1, 1)]), delimiter=",")


if __name__ == "__main__":
    main()
