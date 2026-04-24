"""
SF1546 - Project VT2026
24/3/2026
Group 41

"""

from pathlib import Path

import numpy as np
import scipy.optimize as opt
from utils import NewtonRaphson, NewtonRaphsonResult

ROOT = Path(__file__).parent
plot_dir: Path = ROOT / "Plots"
plot_dir.mkdir(exist_ok=True)
# Results Save directory
results_path: Path = ROOT / "Results"
results_path.mkdir(exist_ok=True)

def main():
    # ============= Setup =============
    Q: float = 801368  # Heat Flux [W]
    Aex: float = 64.15  # Heat transfer area [m^2]
    dTm: float = 29.6  # Mean temp. delta of the two fluids [°C]
    Ds: float = 1.219  # Shell inner diameter [m]
    # Material and Thermodynamic parameters:
    Rfi: float = 1.76e-4  # [m^2·°C/W]
    Rfo: float = 1.76e-4  # [m^2·°C/W]
    hs: float = 356  # [W/m^2·°C]
    ht: float = 356  # [W/m^2·°C]
    kw: float = 60  # [W/m·°C]

    U_f: float = Q / (Aex * dTm)  # Heat transfer coefficient
    d_0: float = 0.007  # Reference initial guess [m]
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

        # Table header for all starting guesses (similar style to before)
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

    print(f"Scalar Newton results written to {results_file}")

if __name__ == "__main__":
    main()
