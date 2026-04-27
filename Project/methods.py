from typing import Callable

import numpy as np

from .structures import (
    EulerResult,
    FixedPointResult,
    NewtonRaphsonResult,
    NewtonSystemResult,
)


def NewtonRaphson(
    f: Callable[[float], float],
    f_prime: Callable[[float], float],
    x0: float,
    tolerance: float,
    max_iterations: int,
) -> NewtonRaphsonResult:
    """
    Newton–Raphson method to find a root of a scalar function f starting from x0.

    Parameters
    ----------
    f : Callable[[float], float]
        Scalar function f(x).
    f_prime : Callable[[float], float]
        Derivative f'(x) of the function f.
    x0 : float
        Initial guess for the root.
    tolerance : float
        Convergence tolerance on |f(x_k)|.
    max_iterations : int
        Maximum allowed number of iterations.

    Returns
    -------
    NewtonRaphsonResult
        Result container with iterates x_k, residuals f(x_k), step sizes Δx_k,
        iteration indices, final iterate, and convergence flag.
    """
    ns: list[int] = []
    fs: list[float] = []
    xs: list[float] = []
    deltas: list[float] = []

    x = x0
    n = 0
    fx = f(x)

    xs.append(x)
    fs.append(fx)
    deltas.append(float("inf"))  # placeholder for initial step
    ns.append(n)

    converged = abs(fx) <= tolerance

    while abs(fx) > tolerance and n < max_iterations:
        n += 1
        df = f_prime(x)
        if df == 0:
            # Derivative zero → Newton step undefined
            break

        x_new = x - fx / df
        delta_x = float(np.abs(x_new - x))
        x = x_new
        fx = f(x)

        xs.append(x)
        fs.append(fx)
        deltas.append(delta_x)
        ns.append(n)

    converged = abs(fx) <= tolerance

    return NewtonRaphsonResult(
        fs=fs,
        xs=xs,
        deltas=deltas,
        ns=ns,
        final_x=x,
        converged=converged,
    )


def fixed_point_iteration(
    g: Callable[[float], float],
    x0: float,
    tolerance: float,
    max_iterations: int,
) -> FixedPointResult:
    """
    Fixed-point iteration to solve x = g(x) starting from x0.

    Parameters
    ----------
    g : Callable[[float], float]
        Iteration function g(x) whose fixed point is sought.
    x0 : float
        Initial guess for the fixed point.
    tolerance : float
        Convergence tolerance on the step size |x_{k+1} - x_k|.
    max_iterations : int
        Maximum allowed number of iterations.

    Returns
    -------
    FixedPointResult
        Result container with iterates x_k, iteration indices, and step sizes Δx_k.

    Raises
    ------
    RuntimeError
        If the method does not converge within max_iterations.
    """
    xs: list[float] = []
    ns: list[int] = []
    deltaX: list[float] = []

    x = x0
    DeltaX = float("inf")
    n = 0

    # Log initial state
    xs.append(x)
    ns.append(n)
    deltaX.append(DeltaX)

    while DeltaX > tolerance and n < max_iterations:
        x_old = x
        x = g(x_old)
        DeltaX = float(np.abs(x - x_old))
        n += 1

        xs.append(x)
        ns.append(n)
        deltaX.append(DeltaX)

    if n >= max_iterations and DeltaX > tolerance:
        raise RuntimeError(
            "Fixed point iteration did not converge within max_iterations."
        )

    return FixedPointResult(xs=xs, ns=ns, deltaX=deltaX)


def euler_forward(
    f: Callable,
    y0: float | np.ndarray,
    domain: list[float],
    h: float,
    *f_args,
    exact: Callable = None,  # type: ignore
) -> EulerResult:
    """
    Solve ODE(s) y' = f(t, y) using the Forward Euler method.

    Supports both scalar ODEs and systems of ODEs.

    Parameters
    ----------
    f : Callable
        Right-hand side of the ODE:
            Scalar: f(t: float, y: float) -> float
            Vector: f(t: float, y: np.ndarray, *f_args) -> np.ndarray
    y0 : float | np.ndarray
        Initial condition. Scalar float or vector np.ndarray([y1_0, ..., yN_0]).
    domain : list[float]
        Time interval [t_start, t_end].
    h : float
        Time step size.
    *f_args
        Additional arguments passed to f(t, y, *f_args) for vector ODEs.
    exact : Callable, optional
        Exact solution y_exact(t). Called as exact(t) with t as np.ndarray.
        Should return an array of the same shape as y.

    Returns
    -------
    EulerResult
        Container with time grid t, numerical solution y, pointwise error e_i,
        and exact solution y_exact (if provided).
    """
    t = np.arange(domain[0], domain[1] + h, h)
    is_scalar: bool = np.isscalar(y0)

    n_vars: int = 1 if is_scalar else len(y0)  # type: ignore
    y: np.ndarray = np.zeros((len(t), n_vars))
    y[0] = y0 if is_scalar else y0.copy()  # type: ignore

    # Time stepping
    for i in range(len(t) - 1):
        f_val = f(t[i], y[i], *f_args) if f_args else f(t[i], y[i])
        y[i + 1] = y[i] + h * f_val

    # Shape output
    if is_scalar:
        y = y[:, 0]  # 1D array for scalar case

    # Error
    if exact is not None:
        y_exact = exact(t)
        e_i = np.abs(y_exact - y)
    else:
        y_exact = None
        e_i = np.zeros_like(y)

    return EulerResult(t=t, y=y, e_i=e_i, y_exact=y_exact)  # type: ignore


def trapets(
    f: Callable[[float], float],
    n: int,
    boundaries: np.ndarray | list[float],
    return_h: bool = False,
) -> float | tuple[float, float]:
    """
    Approximate ∫_a^b f(x) dx using the composite trapezoidal rule.

    Parameters
    ----------
    f : Callable[[float], float]
        Integrand f(x) to be integrated over [a, b].
    n : int
        Number of subintervals.
    boundaries : array_like of float
        Integration limits [a, b].
    return_h : bool, optional
        If True, also return the step size h.

    Returns
    -------
    float or (float, float)
        Approximation of the integral of f from a to b.
        If return_h is True, returns (integral, h).
    """
    if n <= 0:
        raise ValueError("n has to be a positive integer")

    a, b = boundaries
    h = (b - a) / n

    x = np.linspace(a, b, n + 1)
    y = f(x)

    result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    if return_h:
        return result, h
    else:
        return result


def newton_system(
    F_func: Callable,
    JF_func: Callable,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
    args: tuple,
    jac_args: tuple | None = None,
) -> NewtonSystemResult:
    """
    Newton's method for systems of nonlinear equations.

    At each iteration k, solves the linear system
        JF(x_k) * δ = -F(x_k)
    for the correction δ, then updates
        x_{k+1} = x_k + δ,
    until ||F(x_k)||_2 <= tol or the iteration limit is reached.

    Parameters
    ----------
    F_func : Callable
        Vector-valued function F(x, *args) returning np.ndarray of shape (n,).
    JF_func : Callable
        Jacobian function JF(x, *jac_args) returning np.ndarray of shape (n, n).
    x0 : np.ndarray
        Initial guess vector x_0 ∈ ℝ^n (shape (n,)).
    tol : float
        Convergence tolerance on the 2-norm ||F(x_k)||_2.
    max_iter : int
        Maximum allowed number of iterations.
    args : tuple
        Additional parameters passed to F_func.
    jac_args : tuple, optional
        Additional parameters passed to JF_func. Defaults to args if not provided.

    Returns
    -------
    NewtonSystemResult
    """
    jac_args = jac_args if jac_args is not None else args

    x = x0.astype(float)
    xs = [x.copy()]
    Fx = F_func(x, *args)
    Fs = [Fx]
    ns = [0]

    for k in range(max_iter):
        if np.linalg.norm(Fx, ord=2) <= tol:
            break

        J = JF_func(x, *jac_args)
        delta = np.linalg.solve(J, -Fx)
        x = x + delta

        Fx = F_func(x, *args)

        xs.append(x.copy())
        Fs.append(Fx)
        ns.append(k + 1)

    return NewtonSystemResult(xs=xs, Fs=Fs, final_x=x, ns=ns)
