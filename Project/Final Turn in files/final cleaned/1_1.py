from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.optimize as opt

ROOT = Path(__file__).parent
plot_dir: Path = ROOT / "Plots"
plot_dir.mkdir(exist_ok=True)
# Results Save directory
results_path: Path = ROOT / "Results"
results_path.mkdir(exist_ok=True)


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
        Step sizes Δx_k = x_{k+1} - x_k (often useful for convergence diagnostics).
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


def main():
    # ============= Setup =============
    Q: float = 801368  # Heat Flux
    Aex: float = 64.15  # Heat transfer area
    dTm: float = 29.6  # Mean temp. delta of the two fluids
    Ds: float = 1.219  # Shell inner diameter
    # Material and Thermodynamic parameters:
    Rfi: float = 1.76e-4
    Rfo: float = 1.76e-4
    hs: float = 356
    ht: float = 356
    kw: float = 60

    U_f: float = Q / (Aex * dTm)  # Heat transfer coefficient
    d_0: float = 0.007  # Reference initial guess
    tolerance: float = 1e-8

    def g(d) -> float:
        return (
            d / (Ds * ht) + d * Rfi / Ds + d * np.log(d / Ds) / (2 * kw) + Rfo + 1 / hs
        )

    def g_prime(d) -> float:
        return 1 / (Ds * ht) + Rfi / Ds + (np.log(d / Ds) + 1) / (2 * kw)

    def f(d) -> float:
        return 1 / g(d) - U_f

    def f_prime(d) -> float:
        return -g_prime(d) / (g(d) ** 2)

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

    # Run the reference Newton (for summary)
    ref_result: NewtonRaphsonResult = NewtonRaphson(
        f=f, f_prime=f_prime, x0=d_0, tolerance=tolerance, max_iterations=100
    )
    ref_root = ref_result.final_x
    ref_residual = abs(f(ref_root))

    # ============= Results file =============
    results_file = results_path / "Newton Scalar Results.txt"
    with results_file.open("w", encoding="utf-8") as out:
        # Header to match system style
        out.write("Newton's Method for Scalar Nonlinear Equation (d)\n")
        out.write("=" * 100 + "\n\n")

        # Summary for reference run
        out.write("Reference run (starting from d0 = 0.007 m):\n")
        out.write(f"  d*   = {ref_root: .8e} m\n")
        out.write(f"  |f(d*)| = {ref_residual: .3e}\n")
        out.write(f"  Iterations = {len(ref_result.xs) - 1:d}\n\n")

        # Table header for all starting guesses
        out.write(
            f"{'d0 [mm]':>12}  {'d* [mm]':>22}  {'Iterations':>10}  "
            f"{'p (first)':>10}  {'Scipy d* [mm]':>22}  {'|d* - Scipy|':>15}\n"
        )
        out.write("-" * 100 + "\n")

        # ============= Loop over starting guesses =============
        for x0 in starting_vals:
            results: NewtonRaphsonResult = NewtonRaphson(
                f=f, f_prime=f_prime, x0=x0, tolerance=tolerance, max_iterations=100
            )

            # Order of convergence (estimate from error to final_x)
            errors: list[float] = [abs(x - results.final_x) for x in results.xs]
            p: list[float] = [
                np.log(abs(errors[k + 1] / errors[k]))
                / np.log(abs(errors[k] / errors[k - 1]))
                for k in range(1, len(errors) - 1)
                if errors[k] > 0 and errors[k - 1] > 0 and errors[k + 1] > 0
            ]

            # Verification with SciPy
            root, info, exitflag, output = opt.fsolve(
                func=f, x0=x0, fprime=f_prime, full_output=True
            )
            scipy_result = root[0]

            # Write one row for this starting guess
            p_str = f"{p[0]:.6f}" if p else "N/A"
            delta_scipy = abs(results.final_x - scipy_result)
            out.write(
                f"{x0 * 1000:>11.4f}  "  # x0 in mm
                f"{results.final_x * 1000:>22.15f}  "  # d* in mm
                f"{len(results.xs) - 1:>10}  "
                f"{p_str:>10}  "
                f"{scipy_result * 1000:>22.15f}  "  # scipy d* in mm
                f"{delta_scipy:>15.2e}\n"
            )

            # [PLOT_START]
            # ------------- Add to convergence plot -------------
            residuals = [abs(fx) for fx in results.fs]

            is_special = abs(x0 - 0.35350) < 1e-9

            ax_conv.semilogy(
                results.ns,
                residuals,
                linestyle="-",
                color="black" if is_special else None,
                lw=1 if is_special else 0.8,
                marker="o" if is_special else "",
                ms=1.25 if is_special else 1,
                alpha=1.0 if is_special else 0.90,
                zorder=5 if is_special else 2,
                label=rf"$d_0 = {x0 * 1000:.2f}$ mm",
            )
            # [PLOT_END]

    print(f"Scalar Newton results written to {results_file}")

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
    ax1.set_title(r"Root of $f(d)$ - Newton's Method")
    ax1.legend(frameon=False)
    fig1.tight_layout()
    fig1.savefig(plot_dir / "outerdim_newton_roots.png", dpi=300, bbox_inches="tight")

    # ------------- Save convergence plot -------------
    ax_conv.set_xlabel(r"Iteration $k$")
    ax_conv.set_ylabel(r"$|f(d_k)|$")
    ax_conv.set_title("Residual vs Starting Guess - Newton's Method")
    ax_conv.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_conv.legend(frameon=False, fontsize="xx-small", loc="lower left")
    fig_conv.tight_layout()
    fig_conv.savefig(
        plot_dir / "outerdim_newton_convergence.png", dpi=300, bbox_inches="tight"
    )
    # [PLOT_END]


if __name__ == "__main__":
    main()
    plt.show()
