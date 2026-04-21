from typing import Callable
import numpy as np
from .structures import NewtonRaphsonResult, FixedPointResult, EulerResult


def NewtonRaphson(
    f: Callable,
    f_prime: Callable,
    x0: float,
    tolerance: float,
    max_iterations: int,
) -> NewtonRaphsonResult:
    """
    Newton-Raphson method to find a root of f starting from x0.
    Args:
        f: One variable function f(x)
        f_prime: Derivative of f
        x0: initial guess
        tolerance: convergence tolerance on |f(x)|
        max_iterations: maximum number of iterations
    Returns:
        Dataclass containing approximations, residuals, deltas, and iteration counts
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
    deltas.append(float("inf"))
    ns.append(n)

    converged = abs(fx) <= tolerance

    while abs(fx) > tolerance and n < max_iterations:
        n += 1
        df = f_prime(x)
        x_new = x - (fx / df)
        DeltaX = np.abs(x_new - x)
        x = x_new
        fx = f(x)

        fs.append(fx)
        ns.append(n)
        xs.append(x)
        deltas.append(DeltaX)

    converged = abs(fx) <= tolerance

    return NewtonRaphsonResult(
        fs=fs, xs=xs, deltas=deltas, ns=ns, final_x=x, converged=converged
    )

def fixed_point_iteration(
    g: Callable[[float], float],
    x0: float,
    tolerance: float,
    max_iterations: int,
) -> FixedPointResult:
    xs: list[float] = []
    ns: list[int] = []
    deltaX: list[float] = []
    """
    Fixed point iteration to find a fixed point of g starting from x0.
    Args:
        g: function for fixed point iteration
        x0: initial guess
        tolerance: convergence tolerance
        max_iterations: maximum number of iterations
    Returns:
        FixedPointResult containing the list of approximations and iteration counts
    Raises:
        RuntimeError: if the method does not converge within max_iterations

    """

    x = x0
    DeltaX = float('inf')
    n = 0

    while DeltaX > tolerance and n < max_iterations:
        xs.append(x)
        ns.append(n)
        deltaX.append(DeltaX)

        xold = x
        x = g(xold)
        DeltaX = float(np.abs(x - xold))
        n += 1

    if n >= max_iterations and DeltaX > tolerance:
        raise RuntimeError(
            "Fixed point iteration did not converge within max_iterations."
        )

    return FixedPointResult(xs=xs, ns=ns, deltaX=deltaX)

def euler_forward(
    f: Callable,
    y0: float | np.ndarray,
    domain: list,
    h: float,
    *f_args,
    exact: Callable = None,  # type: ignore
) -> EulerResult:
    """
    Solve ODE(s) y' = f(t,y) using Forward Euler method.
    Supports both scalar ODEs f(t,y)→float and systems f(t,y)→np.ndarray.

    Parameters
    ----------
    f : Callable
        ODE right-hand side:

            Scalar: f(t: float, y: float) → float

            Vector: f(t: float, y: np.ndarray, *f_args) → np.ndarray
    y0 : float | np.ndarray
        Initial condition. Scalar float or vector np.ndarray([q0, i0]).
    domain : list[float, float]
        Time interval [t_start, t_end]
    h : float
        Time step size
    *f_args
        Additional arguments passed to f() for vector ODEs (e.g., R, L, C for RLC)
    exact : Callable[[np.ndarray], np.ndarray | float], optional
        Exact solution y(t). Called as exact(t) where t is np.ndarray.

            Scalar: → np.ndarray (n_steps,)

            Vector: → np.ndarray (n_vars, n_steps)


    Returns
    -------
    EulerResult
        Container with t, y (numerical), e_i (errors), y_exact

    Notes
    -----
    - Automatically detects scalar vs vector ODE via np.isscalar(y0)
    - Forward Euler: y_{n+1} = y_n + h * f(t_n, y_n)
    """
    t = np.arange(domain[0], domain[1] + h, h)
    is_scalar: bool = np.isscalar(y0)
    
    n_vars: int = 1 if is_scalar else len(y0)  # type:ignore
    y: np.ndarray = np.zeros((len(t), n_vars))
    y[0] = y0 if is_scalar else y0.copy()  # type:ignore

    # Single step for scalar, vector step for systems
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

    return EulerResult(t=t, y=y, e_i=e_i, y_exact=y_exact)  # type:ignore

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
