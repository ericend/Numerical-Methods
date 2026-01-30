from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def draw(
    f: Callable,
    a: float,
    b: float,
    n: int,
    xlabel: str,
    ylabel: str,
    keypoint_x: float = 0,
    keypoint_y: float = 0,
) -> None:
    """Plots the function f on the interval [a, b] with n points.
    If keypoint_x and keypoint_y are provided, marks that point on the plot.
    """

    x = np.linspace(a, b, n)
    y = f(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)

    # plot settings
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    fig.tight_layout()

    if keypoint_x is not None and keypoint_y is not None:
        plt.plot(keypoint_x, keypoint_y, "ro")  # mark the key point in red

    plt.show()


def main():
    def f(t):
        return 8 * np.exp(-t / 2) * np.cos(3 * t) - 4

    draw(f, 0, 10, 200, "t", "f(t)")


if __name__ == "__main__":
    main()
