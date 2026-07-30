"""Picard iteration for -div((1 + u**2) grad(u)) = f on the unit square."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm, condense, solve
from skfem.helpers import dot, grad


@BilinearForm
def diffusion(u, v, w):
    return (1.0 + w.uk**2) * dot(grad(u), grad(v))


@LinearForm
def load(v, w):
    x, y = w.x
    # Manufactured u = sin(pi x) sin(pi y), f = -div((1+u^2) grad u).
    ue = np.sin(np.pi * x) * np.sin(np.pi * y)
    grad_sq = np.pi**2 * (
        np.cos(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
        + np.sin(np.pi * x) ** 2 * np.cos(np.pi * y) ** 2
    )
    return (2.0 * np.pi**2 * ue * (1.0 + ue**2) - 2.0 * ue * grad_sq) * v


def main() -> None:
    basis = Basis(MeshTri().refined(5), ElementTriP1(), intorder=4)
    boundary = basis.get_dofs().all()
    b = asm(load, basis)
    u = np.zeros(basis.N)
    omega, tolerance = 0.8, 1.0e-10

    for iteration in range(1, 101):
        candidate = solve(*condense(asm(diffusion, basis, uk=basis.interpolate(u)), b, D=boundary))
        update = candidate - u
        u += omega * update
        relative_update = np.linalg.norm(omega * update) / max(np.linalg.norm(u), 1.0)
        print(f"Picard {iteration:2d}: relative update = {relative_update:.3e}")
        if relative_update < tolerance:
            break
    else:
        raise RuntimeError("Picard iteration did not converge")

    x, y = basis.doflocs
    exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    relative_error = np.linalg.norm(u - exact) / np.linalg.norm(exact)
    print(f"DOFs={basis.N}, nodal relative L2 error={relative_error:.3e}")
    assert relative_error < 5.0e-3

    triangulation = mtri.Triangulation(basis.mesh.p[0], basis.mesh.p[1], basis.mesh.t.T)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, values, title in zip(
        axes, (u, exact, np.abs(u - exact)), ("Picard solution", "Exact solution", "Nodal error")
    ):
        image = ax.tricontourf(triangulation, values, levels=20, cmap="viridis")
        fig.colorbar(image, ax=ax)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
