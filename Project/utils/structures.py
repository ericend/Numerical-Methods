import numpy as np
from dataclasses import dataclass


@dataclass
class NewtonRaphsonResult:
    fs: list[float]
    xs: list[float]
    ns: list[int]
    deltas: list[float]
    final_x: float
    converged: bool

@dataclass
class FixedPointResult:
    xs: list[float]
    ns: list[int]
    deltaX: list[float]

@dataclass
class EulerResult:
    """
    Result container for Forward Euler method solutions.

    Fields
    ------
    t : np.ndarray
        Time points where solution is evaluated, shape (n_steps,)
    y : np.ndarray
        Numerical solution. Shape (n_steps,) for scalar ODEs or (n_vars, n_steps) for systems.
    e_i : np.ndarray
        Pointwise absolute error |y_exact(t) - y_euler(t)|. Same shape as y. Zeros if no exact solution provided.
    y_exact : np.ndarray | None
        Exact solution at time points t (if provided). Same shape as y. None otherwise.
    """

    t: np.ndarray
    y: np.ndarray
    e_i: np.ndarray
    y_exact: np.ndarray = None  # type: ignore
