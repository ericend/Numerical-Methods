from typing import Callable

import numpy as np


def trapets(
    f: Callable, n, boundaries: np.ndarray | list, return_h: bool = False
) -> float | tuple:
    """
    Approximates ∫_a^b f(x) dx with the composite trapezoid rule.

    :param f: function to integrate
    :type f: Callable
    :param n: intervals
    :type n: int
    :param boundaries: integration limits [a, b]
    :type boundaries: np.ndarray | list
    :return: Approximation of integral of f from a to b
    :rtype: float
    """

    if n <= 0:
        raise ValueError("n has to be a positive integer")

    a, b = boundaries
    h = (b - a) / n  # Step size

    x = np.linspace(a, b, n + 1)
    y = f(x)

    result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    if return_h:
        return result, h
    else:
        return result
