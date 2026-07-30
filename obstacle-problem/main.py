"""Membrane obstacle problem: penalty, active-set, and semismooth Newton."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat, diags, eye
from scipy.sparse.linalg import spsolve
from skfem import Basis, ElementTriP1, LinearForm, MeshTri, asm
from skfem.models.poisson import laplace


@LinearForm
def downward_load(v, w):
    return -2.0 * v


def obstacle(x, y):
    return 0.16 - 1.8 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 41), np.linspace(0.0, 1.0, 41))
    basis = Basis(mesh, ElementTriP1())
    K = asm(laplace, basis).tocsr()
    load = asm(downward_load, basis)
    fixed = basis.get_dofs().all()
    free = basis.complement_dofs(fixed)
    Kff = K[free][:, free]
    load_free = load[free]
    psi = obstacle(basis.doflocs[0], basis.doflocs[1])
    psi_free = psi[free]
    n = free.size

    penalty_solutions = {}
    penalty_violations = []
    penalty_values = np.asarray((1.0e2, 1.0e3, 1.0e4, 1.0e5))
    penalty_history = None
    for penalty in penalty_values:
        u = spsolve(Kff, load_free)
        history = []
        active_old = None
        for iteration in range(1, 40):
            gap = u - psi_free
            active = gap < 0.0
            residual = Kff @ u - load_free + penalty * np.minimum(gap, 0.0)
            history.append(np.linalg.norm(residual))
            tangent = Kff + diags(penalty * active.astype(float))
            increment = spsolve(tangent, -residual)
            u += increment
            if np.linalg.norm(increment) < 1.0e-12 and np.array_equal(active, active_old):
                break
            active_old = active
        else:
            raise RuntimeError(f"penalty iteration failed for k={penalty}")
        full = np.zeros(basis.N)
        full[free] = u
        penalty_solutions[penalty] = full
        penalty_violations.append(max(0.0, -np.min(u - psi_free)))
        if penalty == penalty_values[-1]:
            penalty_history = history

    # Primal-dual active set: g=u-psi >= 0, lambda >= 0, g*lambda=0.
    u_active = spsolve(Kff, load_free)
    multiplier_active = np.zeros(n)
    active_history = []
    active_set = np.zeros(n, dtype=bool)
    scale = 1.0e3
    for iteration in range(1, 50):
        active_set = multiplier_active - scale * (u_active - psi_free) > 0.0
        inactive = ~active_set
        u_new = psi_free.copy()
        if np.any(inactive):
            u_new[inactive] = spsolve(
                Kff[inactive][:, inactive],
                load_free[inactive] - Kff[inactive][:, active_set] @ psi_free[active_set],
            )
        multiplier_new = np.zeros(n)
        multiplier_new[active_set] = (Kff @ u_new - load_free)[active_set]
        new_active_set = multiplier_new - scale * (u_new - psi_free) > 0.0
        complementarity = np.minimum(multiplier_new, u_new - psi_free)
        active_history.append(np.linalg.norm(complementarity))
        u_active, multiplier_active = u_new, multiplier_new
        if np.array_equal(new_active_set, active_set):
            break
    else:
        raise RuntimeError("active-set iteration failed")

    # Fischer--Burmeister semismooth Newton for stationarity and complementarity.
    u_smooth = spsolve(Kff, load_free)
    multiplier_smooth = np.maximum(Kff @ u_smooth - load_free, 0.0)
    smooth_history = []

    def semismooth_residual(u, multiplier):
        gap = u - psi_free
        stationarity = Kff @ u - load_free - multiplier
        complementarity = np.sqrt(multiplier**2 + gap**2) - multiplier - gap
        return np.concatenate((stationarity, complementarity))

    for iteration_smooth in range(1, 50):
        residual = semismooth_residual(u_smooth, multiplier_smooth)
        residual_norm = np.linalg.norm(residual)
        smooth_history.append(residual_norm)
        if residual_norm < 1.0e-10:
            break
        gap = u_smooth - psi_free
        radius = np.sqrt(multiplier_smooth**2 + gap**2)
        zero = radius < 1.0e-14
        derivative_gap = np.where(zero, -1.0 / np.sqrt(2.0), gap / np.maximum(radius, 1.0e-14) - 1.0)
        derivative_multiplier = np.where(
            zero,
            -1.0 / np.sqrt(2.0),
            multiplier_smooth / np.maximum(radius, 1.0e-14) - 1.0,
        )
        generalized_jacobian = bmat(
            [
                [Kff, -eye(n, format="csr")],
                [diags(derivative_gap), diags(derivative_multiplier)],
            ],
            format="csr",
        )
        increment = spsolve(generalized_jacobian, -residual)
        alpha = 1.0
        while alpha > 1.0e-9:
            trial_u = u_smooth + alpha * increment[:n]
            trial_multiplier = multiplier_smooth + alpha * increment[n:]
            if np.linalg.norm(
                semismooth_residual(trial_u, trial_multiplier)
            ) < residual_norm:
                u_smooth, multiplier_smooth = trial_u, trial_multiplier
                break
            alpha *= 0.5
        else:
            raise RuntimeError("semismooth Newton line search failed")
    else:
        raise RuntimeError("semismooth Newton failed")

    full_active = np.zeros(basis.N)
    full_active[free] = u_active
    full_smooth = np.zeros(basis.N)
    full_smooth[free] = u_smooth
    full_multiplier = np.zeros(basis.N)
    full_multiplier[free] = multiplier_smooth
    relative_difference = np.linalg.norm(u_active - u_smooth) / np.linalg.norm(u_active)
    minimum_gap = np.min(u_smooth - psi_free)
    minimum_multiplier = np.min(multiplier_smooth)
    print(
        f"penalty penetration(k=1e5)={penalty_violations[-1]:.3e}, "
        f"active-set iterations={len(active_history)}, "
        f"semismooth iterations={len(smooth_history)}, "
        f"relative exact-method difference={relative_difference:.3e}, "
        f"min gap={minimum_gap:.3e}, min multiplier={minimum_multiplier:.3e}"
    )
    assert relative_difference < 1.0e-8
    assert minimum_gap > -1.0e-10
    assert minimum_multiplier > -1.0e-10

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    obstacle_image = axes[0, 0].tricontourf(triangulation, psi, levels=24, cmap="terrain")
    fig.colorbar(obstacle_image, ax=axes[0, 0], label=r"obstacle $\psi$")
    axes[0, 0].set(title="Rigid obstacle", xlabel="x", ylabel="y", aspect="equal")

    solution_image = axes[0, 1].tricontourf(
        triangulation, full_smooth, levels=24, cmap="viridis"
    )
    fig.colorbar(solution_image, ax=axes[0, 1], label="membrane displacement")
    axes[0, 1].tricontour(
        triangulation,
        full_smooth - psi,
        levels=[1.0e-8],
        colors="white",
        linewidths=1.5,
    )
    axes[0, 1].set(title="Constrained membrane", xlabel="x", ylabel="y", aspect="equal")

    multiplier_image = axes[0, 2].tricontourf(
        triangulation, np.maximum(full_multiplier, 0.0), levels=24, cmap="magma"
    )
    fig.colorbar(multiplier_image, ax=axes[0, 2], label="contact multiplier")
    axes[0, 2].set(title="Obstacle reaction", xlabel="x", ylabel="y", aspect="equal")

    centre = np.isclose(basis.doflocs[1], 0.5)
    order = np.argsort(basis.doflocs[0, centre])
    x_centre = basis.doflocs[0, centre][order]
    axes[1, 0].plot(x_centre, psi[centre][order], "k--", label="obstacle")
    axes[1, 0].plot(x_centre, full_smooth[centre][order], label="semismooth/PDAS")
    axes[1, 0].plot(
        x_centre,
        penalty_solutions[penalty_values[-1]][centre][order],
        ":",
        label="penalty",
    )
    axes[1, 0].set(title="Centre-line contact", xlabel="x", ylabel="displacement")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].loglog(penalty_values, penalty_violations, "o-")
    axes[1, 1].loglog(
        penalty_values,
        penalty_violations[0] * penalty_values[0] / penalty_values,
        "k--",
        label=r"$O(k_{\rm pen}^{-1})$",
    )
    axes[1, 1].set(title="Penalty consistency", xlabel="penalty", ylabel="maximum penetration")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].semilogy(range(1, len(penalty_history) + 1), penalty_history, "o-", label="penalty")
    axes[1, 2].semilogy(range(1, len(smooth_history) + 1), smooth_history, "s-", label="semismooth Newton")
    axes[1, 2].set(title="Nonlinear convergence", xlabel="iteration", ylabel="residual / merit norm")
    axes[1, 2].grid(True, which="both", alpha=0.3)
    axes[1, 2].legend()
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
