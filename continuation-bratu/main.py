"""Parameter continuation and damped Newton for the two-dimensional Bratu problem."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad


@LinearForm
def residual(v, w):
    return dot(grad(w.uh), grad(v)) - w.lam * np.exp(w.uh) * v


@BilinearForm
def tangent(du, v, w):
    return dot(grad(du), grad(v)) - w.lam * np.exp(w.uh) * du * v


def main() -> None:
    basis = Basis(MeshTri().refined(5), ElementTriP1(), intorder=5)
    fixed = basis.get_dofs().all()
    free = basis.complement_dofs(fixed)
    u = np.zeros(basis.N)
    lambdas, maxima = [], []

    for lam in np.linspace(0.25, 4.0, 16):
        def get_residual(x):
            return asm(residual, basis, uh=basis.interpolate(x), lam=lam)

        for iteration in range(1, 26):
            r = get_residual(u)
            norm_r = np.linalg.norm(r[free])
            if norm_r < 1.0e-10:
                break
            J = asm(tangent, basis, uh=basis.interpolate(u), lam=lam)
            du = np.zeros_like(u)
            du[free] = spsolve(J[free][:, free], -r[free])
            alpha = 1.0
            while alpha > 1.0e-8:
                trial = u + alpha * du
                if np.linalg.norm(get_residual(trial)[free]) < norm_r:
                    u = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(f"line search failed at lambda={lam}")
        else:
            raise RuntimeError(f"Newton failed at lambda={lam}")
        print(
            f"lambda={lam:4.2f}: Newton iterations={iteration:2d}, "
            f"max(u)={u.max():.6f}, residual={norm_r:.3e}"
        )
        lambdas.append(lam)
        maxima.append(u.max())

    assert np.isfinite(u).all() and u.max() > 0.1
    triangulation = mtri.Triangulation(basis.mesh.p[0], basis.mesh.p[1], basis.mesh.t.T)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    axes[0].plot(lambdas, maxima, "o-")
    axes[0].set(title="Bratu continuation path", xlabel=r"$\lambda$", ylabel=r"$\max u$")
    axes[0].grid(True, alpha=0.3)
    image = axes[1].tricontourf(triangulation, u, levels=20, cmap="magma")
    fig.colorbar(image, ax=axes[1], label="u")
    axes[1].set(title=f"Solution at λ={lambdas[-1]:.2f}", xlabel="x", ylabel="y", aspect="equal")
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
