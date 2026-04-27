from pathlib import Path

import numpy as np
from utils.methods import newton_system

# ============= Paths =============
ROOT = Path(__file__).parent
plot_dir: Path = ROOT / "Plots"
plot_dir.mkdir(exist_ok=True)

# ============= Model Definition =============

def make_A(N: int) -> np.ndarray:
    """Tridiagonal (N-1) x (N-1) FD matrix with -2 on diagonal, +1 on off-diagonals."""
    n = N - 1
    return (
        np.diag([-2] * n) + np.diag([1] * (n - 1), k=1) + np.diag([1] * (n - 1), k=-1)
    )

def f(T: float, T_inf: float) -> float:
    return T**4 - T_inf**4

def f_prime(T: float) -> float:
    return 4 * T**3

def interior_grid(N: int, L: float) -> np.ndarray:
    """Interior grid points x_1, ..., x_{N-1}."""
    h = L / N
    return np.linspace(h, L - h, N - 1)

def rhs_vector(
    a: float,
    b: float,
    h: float,
    T_s: float,
    T_L: float,
    T: np.ndarray,
    T_inf: float,
    f: callable,
) -> np.ndarray:
    """RHS vector r(T), same shape as T."""
    rhs = h**2 * (a * (T - T_inf) + b * np.vectorize(f)(T, T_inf))
    rhs[0] -= T_s
    rhs[-1] -= T_L
    return rhs

def F_func(
    T: np.ndarray,
    A: np.ndarray,
    a: float,
    b: float,
    h: float,
    T_s: float,
    T_L: float,
    T_inf: float,
    f: callable,
) -> np.ndarray:
    return A @ T - rhs_vector(a, b, h, T_s, T_L, T, T_inf, f)

def JF_func(
    T: np.ndarray,
    A: np.ndarray,
    a: float,
    b: float,
    h: float,
    f_prime: callable,
) -> np.ndarray:
    d = -(h**2) * (a + b * np.vectorize(f_prime)(T))
    return A + np.diag(d)

def discrete_2_norm(r_vec: np.ndarray) -> float:
    """Discrete 2-norm (eq. 16): sqrt(1/(N-1) * sum |r_k|^2)."""
    return np.sqrt(np.sum(r_vec**2) / (len(r_vec) - 1))

def solve(
    N: int,
    L: float,
    a: float,
    b: float,
    T_s: float,
    T_L: float,
    T_inf: float,
) -> tuple:
    """Run Newton solver and return (result, x_grid)."""
    h = L / N
    A_mat = make_A(N)
    T0 = np.linspace(T_s, T_L, N - 1)
    result = newton_system(
        F_func,
        JF_func,
        T0,
        tol=1e-10,
        max_iter=1000,
        args=(A_mat, a, b, h, T_s, T_L, T_inf, f),
        jac_args=(A_mat, a, b, h, f_prime),
    )
    return result, interior_grid(N, L)

# ============= 8.3 =============

def run_83() -> None:
    # ------------- Parameters (Table 3) -------------
    h_c: float = 40
    K: float = 240
    D: float = 4.13e-3
    T_s: float = 450
    T_inf: float = 293
    T_L: float = T_inf
    L: float = 2.5

    a: float = (4 * h_c) / (D * K)  # alpha_1
    b: float = 0.0  # alpha_2 = 0

    Ns: list[int] = [50 * 2**k for k in range(5)]  # [50, 100, 200, 400, 800]
    N_main: int = 400

    results = {}
    grids = {}
    for N in Ns:
        results[N], grids[N] = solve(N, L, a, b, T_s, T_L, T_inf)

    # ------------- 8.3.a: N=400 vs analytical -------------
    T_num = results[N_main].final_x
    x_grid = grids[N_main]
    T_exact = T_inf + (T_s - T_inf) * np.exp(-np.sqrt(a) * x_grid)
    r_vec = T_num - T_exact
    e_400 = discrete_2_norm(r_vec)

    print("=== 8.3.a ===")
    print(f"Converged in {len(results[N_main].ns) - 1} iterations")
    print(f"Discrete 2-norm ||r||_2 = {e_400:.3e} K")
    print(f"Max pointwise error:      {np.abs(r_vec).max():.3e} K")
    print(f"Error at x=h:             {np.abs(r_vec[0]):.3e} K")
    print(f"Error at x=L-h:           {np.abs(r_vec[-1]):.3e} K")

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x_grid, T_num, lw=1.5, label=f"Numerical (FD, N={N_main})")
    ax1.plot(x_grid, T_exact, lw=1.0, linestyle="--", label="Analytical (eq. 17)")
    ax1.set_xlabel(r"$x$ (m)")
    ax1.set_ylabel(r"$T(x)$ (K)")
    ax1.set_title(
        r"Temperature distribution - numerical vs analytical ($\alpha_2 = 0$)"
    )
    ax1.legend(frameon=False)
    fig1.tight_layout()
    fig1.savefig(plot_dir / "8_3a_solution.png", bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(x_grid, np.abs(r_vec), lw=1.5)
    ax2.set_xlabel(r"$x$ (m)")
    ax2.set_ylabel(r"$|T_{\mathrm{num}} - T_{\mathrm{exact}}|$ (K)")
    ax2.set_title(r"Pointwise error over $[0, L]$")
    fig2.tight_layout()
    fig2.savefig(plot_dir / "8_3a_error.png", bbox_inches="tight")

    # ------------- 8.3.b: Convergence study -------------
    print("\n=== 8.3.b ===")
    norms = []
    for N in Ns:
        T_num_N = results[N].final_x
        T_exact_N = T_inf + (T_s - T_inf) * np.exp(-np.sqrt(a) * grids[N])
        norms.append(discrete_2_norm(T_num_N - T_exact_N))

    # Order of accuracy: eN = Ch^p = C(L/N)^p -> p = log(e_k/e_{k+1}) / log(N_{k+1}/N_k)
    orders = [None] + [
        np.log(norms[k - 1] / norms[k]) / np.log(Ns[k] / Ns[k - 1])
        for k in range(1, len(Ns))
    ]

    print(f"\n{'N':>6}  {'e_N':>12}  {'order p':>10}")
    print("-" * 34)
    for N, e, p in zip(Ns, norms, orders):
        p_str = f"{p:.4f}" if p is not None else "-"
        print(f"{N:>6}  {e:>12.4e}  {p_str:>10}")

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.semilogy(Ns, norms, "o-", lw=1.0, ms=3, label=r"$e_N$")
    ref = norms[0] * (Ns[0] / np.array(Ns)) ** 2
    ax3.semilogy(
        Ns, ref, "--", lw=1, color="crimson", label=r"$\mathcal{O}(N^{-2})$ reference"
    )
    ax3.set_xlabel(r"$N$")
    ax3.set_ylabel(r"$e_N$")
    ax3.set_title(r"Discrete 2-norm error $e_N$ vs $N$")
    ax3.set_xticks(Ns)
    ax3.legend(frameon=False)
    fig3.tight_layout()
    fig3.savefig(plot_dir / "8_3b_convergence.png", bbox_inches="tight")

# ============= 8.4 =============

def run_84() -> None:
    # ------------- Parameters (8.4) -------------
    sigma: float = 5.67e-8
    D: float = 5.0e-3
    T_s: float = 373.15
    T_inf: float = 293.15
    T_L: float = T_inf
    L_84a: float = 0.30  # length for 8.4.a
    N: int = 400

    # ------------- Table 4 -------------
    materials = ["SS AISI 316", "Aluminium (Al)", "Copper (Cu)"]
    Ks: list[float] = [14, 180, 398]
    h_cs: list[float] = [100, 100, 100]
    epss: list[float] = [0.17, 0.82, 0.03]

    # ============= 8.4 Calculations =============
    L_tests: list[float] = [L_84a] + [0.1 + k for k in np.linspace(0, 1, 10)]

    results_84: dict[str, dict[float, np.ndarray]] = {}
    grids_84: dict[str, dict[float, np.ndarray]] = {}

    for mat, K_i, h_c_i, eps_i in zip(materials, Ks, h_cs, epss):
        results_84[mat] = {}
        grids_84[mat] = {}

        a_i = (4 * h_c_i) / (D * K_i)
        b_i = (4 * eps_i * sigma) / (D * K_i)

        for L in L_tests:
            result, x_grid = solve(N, L, a_i, b_i, T_s, T_L, T_inf)
            results_84[mat][L] = result
            grids_84[mat][L] = x_grid

    # ------------- 8.4.a Results -------------
    print("\n=== 8.4.a ===")

    L_first = L_tests[0]

    fig4, ax4 = plt.subplots(figsize=(7, 4))

    for mat in materials:
        result = results_84[mat][L_first]
        x_grid = grids_84[mat][L_first]

        print(f"{mat}: converged in {len(result.ns) - 1} iterations")

        ax4.plot(x_grid, result.final_x, lw=1.5, label=mat)

    ax4.set_xlabel(r"$x$ (m)")
    ax4.set_ylabel(r"$T(x)$ (K)")
    ax4.set_title(r"Temperature distribution for different materials ($N=400$)")
    ax4.legend(frameon=False)
    fig4.tight_layout()
    fig4.savefig(plot_dir / "8_4a_materials.png", bbox_inches="tight")

    # ------------- 8.4.b Results -------------
    tol_frac = 0.01  # e.g. 1% relative error

    print("\n=== 8.4.b ===")

    for mat in materials:
        # Sort lengths to have a clear order
        Ls_sorted = sorted(L_tests)
        L_ref = Ls_sorted[-1]  # longest length as 'infinite' reference

        # Reference solution for this material
        result_ref = results_84[mat][L_ref]
        x_ref = grids_84[mat][L_ref]
        T_ref = result_ref.final_x

        # Compute relative profile error for each shorter length
        rel_errs = {}
        for L in Ls_sorted[:-1]:  # exclude L_ref
            result_L = results_84[mat][L]
            x_L = grids_84[mat][L]
            T_L = result_L.final_x

            # Interpolate reference onto x_L
            T_ref_on_L = np.interp(x_L, x_ref, T_ref)

            diff = T_L - T_ref_on_L
            err = discrete_2_norm(diff)
            ref_norm = discrete_2_norm(T_ref_on_L)
            rel_err = err / ref_norm if ref_norm > 0 else 0.0

            rel_errs[L] = rel_err

        # Pick smallest L with rel_err < tol_frac
        L_min = next((L for L in Ls_sorted[:-1] if rel_errs[L] < tol_frac), None)

        print(f"\n{mat}: reference length L_ref = {L_ref:.2f} m")
        print(f"  {'L (m)':>8}  {'rel_err':>12}  {'converged':>10}")
        print("  " + "-" * 34)
        for L in Ls_sorted[:-1]:
            re = rel_errs[L]
            print(f"  {L:>8.2f}  {re:>12.4e}  {'yes' if re < tol_frac else 'no':>10}")
        print(rf"  -> Estimated L_min \approx {L_min} m")

# ============= Entry Point =============

def main() -> None:
    run_83()
    run_84()

if __name__ == "__main__":
    main()
