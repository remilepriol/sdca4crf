import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

# colors = np.array(list(mcolors.cnames.keys()))
hsv = plt.get_cmap('hsv')


def plot_data(x, y, alpha=1):
    """Plot the two first dimensions of the data."""
    y_range = np.unique(y)
    y_colors = hsv(np.linspace(0, 1, y_range.shape[0] + 1))
    # y_colors = np.random.choice(colors, size=y_range.shape, replace=False,)
    for classe, i in enumerate(y_range):
        # show the points
        plt.scatter(x[y == classe, 0], x[y == classe, 1], c=y_colors[i], label=classe, alpha=alpha)
    plt.legend()
    # set the limits of the plot
    plt.axis('equal')
    plt.autoscale(enable=False)


def discriminator(w, bias=1):
    """Return points on the line of probability 1/2."""
    norm = np.sqrt(np.sum(w[:-1] ** 2))
    linedir = np.array([-w[1], w[0]]) / norm
    line = -w[-1] * bias * w[:2] / norm ** 2 + np.linspace(-1e8, 1e8, 3)[:, np.newaxis] * linedir
    return line


def plot_discriminator(w):
    """Plot the line of probability 1/2."""
    line = discriminator(w)
    plt.plot(line[:, 0], line[:, 1])


def plot_areas(prediction_function, bias, resolution, boundaries=(-4, 4), alpha=.15):
    """Display the prediction of a function defined on R^2. """
    coords = np.linspace(*boundaries, resolution)
    x1, x2 = np.meshgrid(coords, coords)
    x = np.array([x1.flatten(), x2.flatten(), bias * np.ones(resolution ** 2)]).T
    y = prediction_function(x)
    k = np.amax(y) + 1
    # set the number of colors as k+1 to avoid cycling over the colors.
    cmap = mcolors.ListedColormap(hsv(np.linspace(0, 1, k + 1)))
    # make sure every color is present
    y[:k + 1] = np.arange(k + 1)
    plt.pcolormesh(coords, coords, y.reshape([resolution, resolution]), cmap=cmap, alpha=alpha)
