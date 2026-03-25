"""
SF1546 – Project VT2026
24/3/2026
Group 41
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

@dataclass
class NewtonRaphsonResult:
    fs: list[float]
    xs: list[float]
    ns: list[int]
    deltas: list[float]
    final_x: float
    converged: bool

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

def main():
    # -------------- Newton-Raphson -------------------------
    # Known constants
    Q = 801368  # W
    Aex = 64.15  # m^2
    dTm = 29.6  # °C
    Ds = 1.219  # m
    Rfi = 1.76e-4  # m^2°C/W
    Rfo = 1.76e-4  # m^2°C/W
    hs = 356  # W/m^2°C
    ht = 356  # W/m^2°C
    kw = 60  # W/m°C

    C = Q / (Aex * dTm)

    d_0 = 0.007  # m
    tolerance = 1e-8

    def g(d):
        return (
            d / (Ds * ht) + d * Rfi / Ds + d * np.log(d / Ds) / (2 * kw) + Rfo + 1 / hs
        )

    def g_prime(d):
        return 1 / (Ds * ht) + Rfi / Ds + (np.log(d / Ds) + 1) / (2 * kw)

    def f(d):
        return 1 / g(d) - C

    def f_prime(d):
        return -g_prime(d) / g(d) ** 2

    results: NewtonRaphsonResult = NewtonRaphson(
        f=f, f_prime=f_prime, x0=d_0, tolerance=tolerance, max_iterations=100
    )
    print(f"Converged on: {results.final_x}")

    # -------------- Order of Convergence -------------------------
    errors: list[float] = [abs(x - results.final_x) for x in results.xs]
    p: list[float] = [
        np.log(abs(errors[x + 1] / errors[x])) / np.log(abs(errors[x] / errors[x - 1]))
        for x in range(1, len(errors) - 1)
        if errors[x] > 0 and errors[x - 1] > 0 and errors[x + 1]
    ]
    for i, pi in enumerate(p):
        print(f"p_{i + 1} = {pi:.6f}")
        print(f"Iterations: {len(results.xs) - 1}")
    print(f"Errors: {errors}")
    # -------------- Verification -------------------------
    import scipy.optimize as opt

    verification_result: tuple = opt.fsolve(func=f, d_0, fprime=f_prime)
    print(verification_result)

if __name__ == "__main__":
    main()
