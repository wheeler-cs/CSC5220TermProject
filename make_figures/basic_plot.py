"""
Makes a basic plot
"""
import matplotlib.pyplot as plt


# pylint: disable=too-many-arguments,too-many-positional-arguments
def basic_plot(
        x,
        y,
        label: str,
        x_label: str,
        y_label: str,
        title: str,
        filename: str,
        color: str = "blue",
) -> None:
    """
    Makes a basic plot.
    :param x: The x-values to plot.
    :param y: The y-values to plot.
    :param label: The label of the plotted y-values.
    :param x_label: The label for the x-axis.
    :param y_label: The label for the y-axis.
    :param title: The title of the figure.
    :param filename: The filename of the saved figure.
    :param color: The color of the line.
    :return: None.
    """
    # Plot Speed vs. MPG
    plt.figure(figsize=(10, 5))
    plt.plot(
        x,
        y,
        marker='o',
        linestyle='-',
        color=color,
        label=label
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
