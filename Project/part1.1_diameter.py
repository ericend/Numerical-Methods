"""
SF1546 – Project VT2026
24/3/2026
Group 41
"""

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    # [PLOT_START]
    # ============= Plots =============
    plt.style.use("seaborn-v0_8-whitegrid")

    SMALL, MED, BIG = 11, 13, 14
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": MED,
            "axes.titlesize": BIG,
            "axes.labelsize": MED,
            "xtick.labelsize": SMALL,
            "ytick.labelsize": SMALL,
            "legend.fontsize": SMALL,
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # ------------- Plot Function -------------
    fig1, ax1 = plt.subplots(figsize=(6, 4))

    d_plot = np.linspace(0.003, 0.05, 600)
    f_plot = [f(d) for d in d_plot]

    ax1.plot(d_plot * 1000, f_plot, lw=2, label=r"$f(d)$")
    ax1.fill_between(d_plot * 1000, f_plot, alpha=0.07)
    ax1.axhline(0, color="gray", lw=1, ls="--")
    ax1.plot(
        results.final_x * 1000,
        0,
        "o",
        ms=8,
        label=rf"$d^* = {results.final_x * 1000:.2f}$ mm",
    )
    ax1.annotate(
        rf"$d^* = {results.final_x * 1000:.2f}$ mm",
        xy=(results.final_x * 1000, 0),
        xytext=(results.final_x * 1000 - 8, 20),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=SMALL,
    )
    ax1.set_xlabel(r"$d$ (mm)")
    ax1.set_ylabel(r"$f(d)$")
    ax1.set_title(r"Root of $f(d)$ — Newton's Method")
    ax1.legend(frameon=False)
    fig1.tight_layout()
    fig1.savefig("plot_f.png", dpi=300, bbox_inches="tight")

    # ------------- Plot Convergence -------------
    residuals = [abs(fx) for fx in results.fs]

    fig2, ax2 = plt.subplots(figsize=(5, 4))

    ax2.semilogy(results.ns, residuals, "o-", lw=2, ms=8, label=r"$r_k = |f(d_k)|$")
    ax2.semilogy(results.ns, errors, "s--", lw=2, ms=8, label=r"$e_k = |d_k - d^*|$")
    ax2.set_xlabel(r"Iteration $k$")
    ax2.set_ylabel(r"Magnitude")
    ax2.set_title("Residual and Error — Newton's Method")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig("plot_convergence.png", dpi=300, bbox_inches="tight")

    # [PLOT_END]


if __name__ == "__main__":
    main()
