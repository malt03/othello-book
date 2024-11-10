import numpy as np
import matplotlib.pyplot as plt


def main():
    points = np.random.rand(200, 2)
    labels = np.zeros(200)

    # より短い周期の複雑な曲線境界の生成
    for i in range(200):
        # 周波数を高くして短い周期の曲線境界
        boundary = (
            0.3 * np.sin(10 * points[i, 0])
            + 0.2 * np.cos(15 * points[i, 0])
            + 0.1 * np.sin(20 * points[i, 0])
            + 0.5
            + np.random.normal(0, 0.02)
        )

        # 基本のラベル付け（境界より上なら1、下なら0）
        if points[i, 1] > boundary:
            labels[i] = 1
            # 境界に近いポイントにはランダム性を追加
            if np.abs(points[i, 1] - boundary) < 0.1:
                labels[i] = np.random.choice([0, 1], p=[0.3, 0.7])  # 30%の確率で0に
        else:
            if np.abs(points[i, 1] - boundary) < 0.1:
                labels[i] = np.random.choice([0, 1], p=[0.7, 0.3])  # 30%の確率で1に

    save_csv(points, labels, "data.csv")

    # プロット
    red = points[labels == 0]
    blue = points[labels == 1]
    plt.scatter(red[:, 0], red[:, 1], c="red", label="Class 0")
    plt.scatter(blue[:, 0], blue[:, 1], c="blue", label="Class 1")
    plt.show()


def save_csv(points, labels, filename):
    np.savetxt(filename, np.hstack([points, labels.reshape(-1, 1)]), delimiter=",")


if __name__ == "__main__":
    main()
