"""Utility functions"""
from matplotlib import pyplot as plt


def plot_batch(images, titles, figsize=None):
    """Plot a batch of images, two on a line"""
    lines = (len(images) + 1) // 2

    plt.interactive(True)
    fig, axes = plt.subplots(lines, 2, figsize=figsize)

    if lines == 1:
        axes = [axes]

    for l in range(lines):
        for c in range(2):
            axes[l][c].imshow(images[2 * l + c], cmap='gray')
            axes[l][c].set_title(titles[2 * l + c])

    return fig
