import torchvision
import matplotlib.pyplot as plt


def main():
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)

    num_images = 100
    images_and_labels = [(dataset[i][0], dataset[i][1]) for i in range(num_images)]

    cols = 10
    rows = (num_images + cols - 1) // cols

    _fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    for idx, (image, label) in enumerate(images_and_labels):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_title(f"{label}")
        ax.axis("off")

    for idx in range(num_images, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    main()
