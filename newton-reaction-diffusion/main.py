"""Damped Newton method for -Laplacian(u) + u**3 = f on the unit square."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad


@LinearForm
def residual(v, w):
    return dot(grad(w.uh), grad(v)) + w.uh**3 * v - w.f * v


@BilinearForm
def tangent(du, v, w):
    return dot(grad(du), grad(v)) + 3.0 * w.uh**2 * du * v


def forcing(x):
    ue = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return 2.0 * np.pi**2 * ue + ue**3


def main() -> None:
    basis = Basis(MeshTri().refined(5), ElementTriP1(), intorder=5)
    fixed = basis.get_dofs().all()
    free = basis.complement_dofs(fixed)
    u = np.zeros(basis.N)
    f_qp = forcing(basis.global_coordinates())
    residual_history = []

    def get_residual(x):
        return asm(residual, basis, uh=basis.interpolate(x), f=f_qp)

    initial_norm = None
    for iteration in range(1, 21):
        r = get_residual(u)
        norm_r = np.linalg.norm(r[free])
        residual_history.append(norm_r)
        initial_norm = norm_r if initial_norm is None else initial_norm
        print(f"Newton {iteration:2d}: residual = {norm_r:.3e}")
        if norm_r < max(1.0e-11, 1.0e-10 * initial_norm):
            break
        J = asm(tangent, basis, uh=basis.interpolate(u))
        du = np.zeros_like(u)
        du[free] = spsolve(J[free][:, free], -r[free])

        alpha = 1.0
        while alpha > 1.0e-8:
            trial = u + alpha * du
            if np.linalg.norm(get_residual(trial)[free]) <= (1.0 - 1.0e-4 * alpha) * norm_r:
                u = trial
                break
            alpha *= 0.5
        else:
            raise RuntimeError("Newton line search failed")
    else:
        raise RuntimeError("Newton iteration did not converge")

    x, y = basis.doflocs
    exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    relative_error = np.linalg.norm(u - exact) / np.linalg.norm(exact)
    print(f"DOFs={basis.N}, nodal relative L2 error={relative_error:.3e}")
    assert relative_error < 3.0e-3

    triangulation = mtri.Triangulation(basis.mesh.p[0], basis.mesh.p[1], basis.mesh.t.T)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    image = axes[0].tricontourf(triangulation, u, levels=20, cmap="viridis")
    fig.colorbar(image, ax=axes[0], label="u")
    axes[0].set(title="Nonlinear reaction–diffusion", xlabel="x", ylabel="y", aspect="equal")
    axes[1].semilogy(range(1, len(residual_history) + 1), residual_history, "o-")
    axes[1].set(title="Newton convergence", xlabel="iteration", ylabel="residual norm")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
