import matplotlib.colors
import matplotlib.pyplot as plt
import scipy


def plot_labels(lbl: scipy.ndarray, lbl_count: int) -> None:
    """Shows a plot of the given label image with a random color map."""
    color_map = scipy.rand(lbl_count, 3)
    color_map = matplotlib.colors.ListedColormap(color_map)
    plt.imshow(lbl, cmap=color_map)
    plt.show()
