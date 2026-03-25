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
import scipy.optimize as opt


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
    # ============= Setup =============
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

    starting_vals: np.ndarray = np.linspace(d_0, 0.7, 13)

    # [PLOT_START]
    # ============= Plot Setup =============
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
        }
    )

    fig_conv, ax_conv = plt.subplots(figsize=(6, 4))
    # [PLOT_END]

    # ============= Loop =============
    with open("results.txt", "w") as out:
        out.write(
            f"{'x0':>12}  {'d*':>22}  {'Iterations':>10}  {'p':>10}  {'Scipy d*':>22}  {'|d* - Scipy|':>22}\n"
        )
        out.write("-" * 109 + "\n")

        for x0 in starting_vals:
            results: NewtonRaphsonResult = NewtonRaphson(
                f=f, f_prime=f_prime, x0=x0, tolerance=tolerance, max_iterations=100
            )

            # ------------- Order of Convergence -------------
            errors: list[float] = [abs(x - results.final_x) for x in results.xs]
            p: list[float] = [
                np.log(abs(errors[k + 1] / errors[k]))
                / np.log(abs(errors[k] / errors[k - 1]))
                for k in range(1, len(errors) - 1)
                if errors[k] > 0 and errors[k - 1] > 0 and errors[k + 1] > 0
            ]

            # ------------- Verification -------------
            scipy_result = opt.fsolve(func=f, x0=x0, fprime=f_prime)[0]

            # ------------- Write to file -------------
            p_str = f"{p[0]:.6f}" if p else "N/A"
            delta_scipy = abs(results.final_x - scipy_result)
            out.write(
                f"{x0 * 1000:>11.4f}mm  "
                f"{results.final_x:>22.15f}  "
                f"{len(results.xs) - 1:>10}  "
                f"{p_str:>10}  "
                f"{scipy_result:>22.15f}  "
                f"{delta_scipy:>22.2e}\n"
            )
            # [PLOT_START]
            # ------------- Add to convergence plot -------------
            residuals = [abs(fx) for fx in results.fs]
            ax_conv.semilogy(
                results.ns,
                residuals,
                "o-",
                lw=1.5,
                ms=5,
                label=rf"$d_0 = {x0 * 1000:.2f}$ mm",
            )
            # [PLOT_END]

    # [PLOT_START]
    # ------------- Plot f(d) -------------
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    d_plot = np.linspace(0.003, 1, 600)
    f_plot = [f(d) for d in d_plot]
    final_result = NewtonRaphson(
        f=f, f_prime=f_prime, x0=d_0, tolerance=tolerance, max_iterations=100
    )

    ax1.plot(d_plot * 1000, f_plot, lw=2, label=r"$f(d)$")
    ax1.fill_between(d_plot * 1000, f_plot, alpha=0.07)
    ax1.axhline(0, color="gray", lw=1, ls="--")
    ax1.plot(
        final_result.final_x * 1000,
        0,
        "o",
        ms=8,
        label=rf"$d_1^* = {final_result.final_x * 1000:.2f}$ mm",
    )
    ax1.annotate(
        rf"$d_1^* = {final_result.final_x * 1000:.2f}$ mm",
        xy=(final_result.final_x * 1000, 0),
        xytext=(final_result.final_x * 1000 + 100, -300),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=SMALL,
    )
    ax1.plot(
        results.final_x * 1000,
        0,
        "o",
        ms=8,
        label=rf"$d_2^* = {results.final_x * 1000:.2f}$ mm",
    )
    ax1.annotate(
        rf"$d_2^* = {results.final_x * 1000:.2f}$ mm",
        xy=(results.final_x * 1000, 0),
        xytext=(results.final_x * 1000 - 370, -300),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=SMALL,
    )
    ax1.set_xlabel(r"$d$ (mm)")
    ax1.set_ylabel(r"$f(d)$")
    ax1.set_title(r"Root of $f(d)$ — Newton's Method")
    ax1.legend(frameon=False)
    fig1.tight_layout()
    fig1.savefig("plot_f.png", dpi=300, bbox_inches="tight")

    # ------------- Save convergence plot -------------
    ax_conv.set_xlabel(r"Iteration $k$")
    ax_conv.set_ylabel(r"$|f(d_k)|$")
    ax_conv.set_title("Residual vs Starting Guess — Newton's Method")
    ax_conv.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_conv.legend(frameon=False, fontsize=9)
    fig_conv.tight_layout()
    fig_conv.savefig("plot_convergence.png", dpi=300, bbox_inches="tight")
    # [PLOT_END]


if __name__ == "__main__":
    main()
