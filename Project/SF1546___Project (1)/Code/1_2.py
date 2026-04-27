from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Plot save directory
ROOT = Path(__file__).parent
plot_dir: Path = ROOT / "Plots"
plot_dir.mkdir(exist_ok=True)
# Results Save directory
results_path: Path = ROOT / "Results"
results_path.mkdir(exist_ok=True)


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
        JF(x_k) * delta = -F(x_k)
    for the correction delta, then updates
        x_{k+1} = x_k + delta,
    until ||F(x_k)||_2 <= tol or the iteration limit is reached.

    Parameters
    ----------
    F_func : Callable
        Vector-valued function F(x, *args) returning np.ndarray of shape (n,).
    JF_func : Callable
        Jacobian function JF(x, *jac_args) returning np.ndarray of shape (n, n).
    x0 : np.ndarray
        Initial guess vector x_0  in  R^n (shape (n,)).
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


# ============= Model Definition =============


def g(
    d: float, Ds: float, ht: float, Rfi: float, kw: float, Rfo: float, hs: float
) -> float:
    """
    g(d, Ds) from the report:
        g(d, D_s) = d/(D_s h_t) + d R_fi / D_s + d ln(d/D_s)/(2 k_w) + R_fo + 1/h_s
    """
    return (
        d / (Ds * ht) + d * Rfi / Ds + d * np.log(d / Ds) / (2.0 * kw) + Rfo + 1.0 / hs
    )


def g_d(d: float, Ds: float, ht: float, Rfi: float, kw: float) -> float:
    """
    dg/dd from the report:
        g_d = 1/(D_s h_t) + R_fi / D_s + (ln(d/D_s) + 1)/(2 k_w)
    """
    return 1.0 / (Ds * ht) + Rfi / Ds + (np.log(d / Ds) + 1.0) / (2.0 * kw)


def g_Ds(d: float, Ds: float, ht: float, Rfi: float, kw: float) -> float:
    """
    dg/dD_s from the report:
        g_{D_s} = -d / (D_s^2 h_t) - d R_fi / D_s^2 - d/(2 k_w D_s)
    """
    return -d / (Ds**2 * ht) - d * Rfi / (Ds**2) - d / (2.0 * kw * Ds)


# Newton Function vector
def F(
    x: np.ndarray,
    Q: float,
    dTm: float,
    K1: float,
    n1: float,
    c: float,
    S_t: float,
    hs: float,
    ht: float,
    Rfi: float,
    Rfo: float,
    kw: float,
    dP_max: float,
) -> np.ndarray:
    """
    Vector function F(d, Ds) = (F1, F2)^T corresponding to eqs. (F1) and (F2) in eq.(11) in report.

    F1(d, Ds) = g(d, Ds)^(-1) - Q d^{n1-1} / (pi K1 Ds^{n1} DeltaT_m)
    F2(d, Ds) = c / (Ds^2 (S_t - d)^2) - DeltaP_max
    """
    d, Ds = x
    g_val = g(d, Ds, ht, Rfi, kw, Rfo, hs)

    F1 = (1.0 / g_val) - Q * d ** (n1 - 1.0) / (np.pi * K1 * Ds**n1 * dTm)
    F2 = c / (Ds**2 * (S_t - d) ** 2) - dP_max

    return np.array([F1, F2], dtype=float)


# Jacobian for Newton Function vector
def JF(
    x: np.ndarray,
    Q: float,
    dTm: float,
    K1: float,
    n1: float,
    c: float,
    S_t: float,
    hs: float,
    ht: float,
    Rfi: float,
    Rfo: float,
    kw: float,
    dP_max: float,
) -> np.ndarray:
    """
    Jacobian matrix JF(d, Ds) corresponding to eq. (12) in report:

        JF = [[dF1/dd,  dF1/dDs],
              [dF2/dd,  dF2/dDs]]
    """
    d, Ds = x
    g_val = g(d, Ds, ht, Rfi, kw, Rfo, hs)
    gd = g_d(d, Ds, ht, Rfi, kw)
    gDs = g_Ds(d, Ds, ht, Rfi, kw)

    # dF1/dd
    J11 = -gd / (g_val**2) + (n1 - 1.0) * Q * (d ** (n1 - 2.0)) / (
        np.pi * K1 * Ds**n1 * dTm
    )

    # dF1/dDs
    J12 = gDs / (g_val**2) + n1 * Q * d ** (n1 - 1.0) / (
        np.pi * K1 * Ds ** (n1 + 1.0) * dTm
    )

    # dF2/dd
    J21 = 2.0 * c / (Ds**2 * (S_t - d) ** 3)

    # dF2/dDs
    J22 = -2.0 * c / (Ds**3 * (S_t - d) ** 2)

    return np.array([[J11, J12], [J21, J22]], dtype=float)


# ============= Main  =============


def main() -> None:
    # ------------- Parameters -------------
    # Heat / thermo parameters
    Q: float = 801_368.0  # Heat flux
    dTm: float = 29.6  # Mean temp. difference
    dP_max: float = 49_080.0  # Max allowed pressure drop

    # Material parameters
    c: float = 0.389
    S_t: float = 0.016
    K_1: float = 0.249
    n_1: float = 2.207

    # Thermal parameters / coefficients
    Rfi: float = 1.76e-4
    Rfo: float = 1.76e-4
    hs: float = 356.0
    ht: float = 356.0
    kw: float = 60.0

    # Newton settings
    tol: float = 1e-8
    max_iter: int = 50

    # Initial guess for (d, Ds)
    d0: float = 0.015
    Ds0: float = 0.8
    x0 = np.array([d0, Ds0], dtype=float)

    # Pack parameters in the order F, JF expect
    params = (Q, dTm, K_1, n_1, c, S_t, hs, ht, Rfi, Rfo, kw, dP_max)

    # ============= Newton solve =============
    result = newton_system(
        F_func=F,
        JF_func=JF,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
        args=params,
    )

    d_star, Ds_star = result.final_x
    residual_norm = np.linalg.norm(result.Fs[-1], ord=2)

    print(f"Converged in {len(result.ns) - 1} iterations")
    print(f"d*  = {d_star:.6e} m")
    print(f"Ds* = {Ds_star:.6e} m")
    print(f"||F(x*)||_2 = {residual_norm:.3e}")

    # ============= Convergence analysis =============

    # ------------- Per-iteration residuals and errors -------------
    residuals = [np.linalg.norm(Fk, ord=2) for Fk in result.Fs]
    x_star = result.final_x
    errors = [np.linalg.norm(x_k - x_star, ord=2) for x_k in result.xs]

    # Pairs of e_k, e_k+1 for error plot
    e_k = np.array(errors[:-1])
    e_kp1 = np.array(errors[1:])

    # ------------- Per-iteration Jacobian determinant and condition number -------------
    det_Js: list[float] = []
    cond_Js: list[float] = []
    for x_k in result.xs:
        J_k = JF(x_k, *params)
        det_Js.append(np.linalg.det(J_k))
        cond_Js.append(np.linalg.cond(J_k))

    # ------------- Per-iteration order p_k (aligned with k) -------------
    p_vals: list[float | None] = [None] * len(errors)
    for k in range(1, len(errors) - 1):
        e_km1, e_k, e_kp1 = errors[k - 1], errors[k], errors[k + 1]
        if e_km1 > 0 and e_k > 0 and e_kp1 > 0:
            p_k = np.log(e_kp1 / e_k) / np.log(e_k / e_km1)
            p_vals[k] = p_k

    # Jacobian at final iterate (for summary)
    J_star = JF(x_star, *params)
    det_J = np.linalg.det(J_star)
    cond_J = np.linalg.cond(J_star)

    # ============= Write Results to .txt file =============
    results_file = results_path / "Newton System Results.txt"
    with results_file.open("w", encoding="utf-8") as out:
        out.write("Newton's Method for Nonlinear System (d, D_s)\n")
        out.write("=" * 80 + "\n\n")
        out.write("Initial Guess:\n")
        out.write(f"d_0 = {d0}\n")
        out.write(f"D_s0 = {Ds0}\n\n")
        out.write("Final solution:\n")
        out.write(f"  d*   = {d_star: .8e} m\n")
        out.write(f"  D_s* = {Ds_star: .8e} m\n")
        out.write(f"  ||F(x*)||_2 = {residual_norm: .3e}\n")
        out.write(f"  Iterations  = {len(result.ns) - 1:d}\n\n")

        out.write("Jacobian at x*:\n")
        out.write(f"  det(J(x*))  = {det_J: .8e}\n")
        out.write(f"  cond(J(x*)) = {cond_J: .3e}\n\n")

        out.write("Per-iteration data:\n")
        out.write(
            f"{'k':>4}  {'d_k [mm]':>12}  {'D_s,k [mm]':>12}  "
            f"{'||F(x_k)||_2':>18}  {'||x_k - x*||_2':>18}  "
            f"{'det(J_k)':>12}  {'cond(J_k)':>12}  {'p_k':>12}\n"
        )
        out.write("-" * 120 + "\n")

        for k, (x_k, res_k, err_k, det_k, cond_k, p_k) in enumerate(
            zip(result.xs, residuals, errors, det_Js, cond_Js, p_vals)
        ):
            d_k, Ds_k = x_k
            if p_k is not None:
                out.write(
                    f"{k:4d}  {d_k * 1000:12.6e}  {Ds_k * 1000:12.6e}  "
                    f"{res_k:18.8e}  {err_k:18.8e}  "
                    f"{det_k:12.4e}  {cond_k:12.3e}  {p_k:12.6f}\n"
                )
            else:
                out.write(
                    f"{k:4d}  {d_k * 1000:12.6e}  {Ds_k * 1000:12.6e}  "
                    f"{res_k:18.8e}  {err_k:18.8e}  "
                    f"{det_k:12.4e}  {cond_k:12.3e}  {'-':>12}\n"
                )

    print(f"Results written to {results_file}")

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

    # ========= 1) Residual norm vs iteration (semilogy) =========
    fig_res, ax_res = plt.subplots(figsize=(6, 4))
    residuals = [np.linalg.norm(Fk, ord=2) for Fk in result.Fs]

    ax_res.semilogy(result.ns, residuals, "o-", lw=1.5, ms=5, label="System Newton")
    ax_res.set_xlabel(r"Iteration $k$")
    ax_res.set_ylabel(r"$\|F(x_k)\|_2$")
    ax_res.set_title(r"Residual norm vs iteration - Newton's method (system)")
    ax_res.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_res.legend(frameon=False)
    fig_res.tight_layout()
    fig_res.savefig(
        plot_dir / "system_newton_residuals.png",
        dpi=300,
        bbox_inches="tight",
    )

    # ========= 2) Error norm vs iteration (semilogy) =========
    fig_err, ax_err = plt.subplots(figsize=(6, 4))
    x_star = result.final_x
    errors = [np.linalg.norm(x_k - x_star, ord=2) for x_k in result.xs]

    ax_err.semilogy(
        result.ns,
        errors,
        "o-",
        lw=1.5,
        ms=5,
        label=r"$e_k = \|x_k - x^*\|_2$",
    )
    ax_err.set_xlabel(r"Iteration $k$")
    ax_err.set_ylabel(r"$\|x_k - x^*\|_2$")
    ax_err.set_title(r"Error norm vs iteration - Newton's method (system)")
    ax_err.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_err.legend(frameon=False)
    fig_err.tight_layout()
    fig_err.savefig(
        plot_dir / "system_newton_errors.png",
        dpi=300,
        bbox_inches="tight",
    )

    # ========= Log-log plot of e_{k+1} vs e_k =========
    fig_ek, ax_ek = plt.subplots(figsize=(6, 4))
    e_k = np.array(errors[:-1])
    e_kp1 = np.array(errors[1:])

    ax_ek.loglog(
        e_k,
        e_kp1,
        "o-",
        lw=1.5,
        ms=5,
        label=r"Newton iterates",
    )
    ax_ek.set_xlabel(r"$e_k = \|x_k - x^*\|_2$")
    ax_ek.set_ylabel(r"$e_{k+1} = \|x_{k+1} - x^*\|_2$")
    ax_ek.set_title(r"Log-log plot of $e_{k+1}$ versus $e_k$")

    # Reference slopes p = 1 and p = 2 in the log-log plot
    ref_x = e_k[len(e_k) // 2]  # pick a mid error as reference (must be > 0)

    for p_ref, color, label in [
        (1.0, "gray", r"Reference $e_{k+1} \propto e_k^1$"),
        (2.0, "black", r"Reference $e_{k+1} \propto e_k^2$"),
    ]:
        ref_y = ref_x**p_ref
        x_line = np.array([ref_x / 10, ref_x * 10])
        y_line = ref_y * (x_line / ref_x) ** p_ref
        ax_ek.loglog(
            x_line,
            y_line,
            "--",
            color=color,
            lw=1,
            label=label,
        )

    ax_ek.legend(frameon=False)
    fig_ek.tight_layout()
    fig_ek.savefig(
        plot_dir / "system_newton_loglog_errors.png",
        dpi=300,
        bbox_inches="tight",
    )

    # ========= Newton Path  =========
    fig_path, ax_path = plt.subplots(figsize=(6, 4))
    d_vals = [x[0] for x in result.xs]
    Ds_vals = [x[1] for x in result.xs]

    ax_path.plot(Ds_vals, d_vals, "o-", ms=5, lw=1.5, label="Newton path")
    ax_path.plot(Ds_vals[0], d_vals[0], "s", ms=7, label="Initial guess")
    ax_path.plot(Ds_vals[-1], d_vals[-1], "o", ms=7, label="Final iterate")

    ax_path.set_xlabel(r"$D_s$ (m)")
    ax_path.set_ylabel(r"$d$ (m)")
    ax_path.set_title(r"Newton iteration path in $(D_s, d)$-plane")
    ax_path.legend(frameon=False)
    fig_path.tight_layout()
    fig_path.savefig(
        plot_dir / "system_newton_path.png",
        dpi=300,
        bbox_inches="tight",
    )
    # [PLOT_END]


if __name__ == "__main__":
    main()
    plt.show()
