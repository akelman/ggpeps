import numpy as np
import matplotlib.pyplot as plt


def calculate_gradient_2d(x, y, data):
    dx = np.diff(x)
    dy = np.diff(y)
    diff_data_x = np.diff(data, axis=1)
    diff_data_y = np.diff(data, axis=0)
    dx = np.tile(dx, [diff_data_x.shape[0], 1])
    dy = np.tile(np.reshape(dy, (-1, 1)), [1, diff_data_y.shape[1]])
    diff_data_x /= dx
    diff_data_y /= dy
    return diff_data_x, diff_data_y


def grad_heatmap(X, Y, data, cbar_kw={}, cbarlabel="", ax=None, **kwargs):
    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(["%.2f" % x for x in X], rotation=90)
    ax.set_yticklabels(["%.2f" % y for y in Y])

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Both axis are increasing
    x_lim = ax.set_xlim(sorted(ax.get_xlim()))
    y_lim = ax.set_ylim(sorted(ax.get_ylim()))
    x_scale = (x_lim[1] - 0.5 - (x_lim[0] + 0.5)) / (max(X) - min(X))
    y_scale = (y_lim[1] - 0.5 - (y_lim[0] + 0.5)) / (max(Y) - min(Y))
    # Calculate the differential
    diff_data_x, diff_data_y = calculate_gradient_2d(X, Y, data)
    # Plot gradient for x
    quiver_x, quiver_y = np.meshgrid(X[:-1], Y)
    plt.quiver(
        (quiver_x) * x_scale + 0.5,
        (quiver_y) * y_scale,
        diff_data_x * x_scale,
        0,
        color="w",
    )
    # Plot gradient for y
    quiver_x, quiver_y = np.meshgrid(X, Y[:-1])
    plt.quiver(
        (quiver_x) * x_scale,
        (quiver_y) * y_scale + 0.5,
        0,
        diff_data_y * y_scale,
        color="k",
    )
    return im, cbar


if __name__ == "__main__":
    X = np.array([0.0, 1.0, 2.0])
    Y = np.array([0.0, 2.0, 4.0])
    data = np.array([[3.0, 4.0, 7.0], [8.0, 12.0, 13.0], [3.0, 5.0, 2.0]])
    im, cbar = grad_heatmap(X, Y, data, cmap=plt.get_cmap("viridis"))
    plt.show()
