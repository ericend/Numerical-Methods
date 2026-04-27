from dataclasses import dataclass

import numpy as np

@dataclass
class NewtonRaphsonResult:
    """
    Result container for scalar Newton-Raphson iterations.

    Parameters
    ------
    fs : list[float]
        Sequence of function values f(x_k) at each iteration k.
    xs : list[float]
        Sequence of iterates x_k produced by Newton's method.
    ns : list[int]
        Iteration indices corresponding to each x_k (typically [0, 1, ..., n]).
    deltas : list[float]
        Step sizes Deltax_k = x_{k+1} - x_k (often useful for convergence diagnostics).
    final_x : float
        Final iterate x_* returned by the method (last element of xs).
    converged : bool
        Flag indicating whether the stopping criterion was satisfied.
    """

    fs: list[float]
    xs: list[float]
    ns: list[int]
    deltas: list[float]
    final_x: float
    converged: bool

@dataclass
class NewtonSystemResult:
    """
    Result container for Newton's method applied to systems of nonlinear equations.

    Parameters
    ------
    xs : list[np.ndarray]
        Sequence of iterates x_k  in  R^n at each step (shape (n,) per iterate).
    Fs : list[np.ndarray]
        Sequence of residual vectors F(x_k)  in  R^n at each iteration.
    final_x : np.ndarray
        Final iterate x_*  in  R^n returned by the method (last element of xs).
    ns : list[int]
        Iteration indices corresponding to each x_k (typically [0, 1, ..., n]).
    """

    xs: list[np.ndarray]  # iterates x_k
    Fs: list[np.ndarray]  # F(x_k)
    final_x: np.ndarray  # x_* at termination
    ns: list[int]  # iteration indices

@dataclass
class FixedPointResult:
    """
    Result container for fixed-point iteration x_{k+1} = g(x_k).

    Parameters
    ------
    xs : list[float]
        Sequence of iterates x_k produced by the fixed-point iteration.
    ns : list[int]
        Iteration indices corresponding to each x_k (typically [0, 1, ..., n]).
    deltaX : list[float]
        Step sizes Deltax_k = x_{k+1} - x_k, useful for monitoring convergence.
    """

    xs: list[float]
    ns: list[int]
    deltaX: list[float]

@dataclass
class EulerResult:
    """
    Result container for Forward Euler method solutions.

    Parameters
    ------
    t : np.ndarray
        Time points where the solution is evaluated, shape (n_steps,).
    y : np.ndarray
        Numerical solution. Shape (n_steps,) for scalar ODEs or (n_vars, n_steps)
        for systems of ODEs.
    e_i : np.ndarray
        Pointwise absolute error |y_exact(t) - y_euler(t)|. Same shape as y.
        Filled with zeros if no exact solution is provided.
    y_exact : np.ndarray | None
        Exact solution evaluated at time points t (if available). Same shape as y.
        None otherwise.
    """

    t: np.ndarray
    y: np.ndarray
    e_i: np.ndarray
    y_exact: np.ndarray = None
