"""Semi-implicit time stepping for the Allen--Cahn equation on a unit square."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad


@BilinearForm
def mass(u, v, w):
    return u * v


@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def cubic_linearization(u, v, w):
    # Linearization (u^n)^2 u^(n+1) of the cubic term.
    return w.un**2 * u * v


@LinearForm
def right_hand_side(v, w):
    return (w.un / w.dt + w.un) * v


def main() -> None:
    basis = Basis(MeshTri().refined(5), ElementTriP1(), intorder=4)
    x, y = basis.doflocs
    u = 0.35 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    initial_u = u.copy()
    dt, epsilon, steps = 2.0e-3, 0.04, 250
    M = asm(mass, basis)
    K = asm(stiffness, basis)

    def energy(xvec):
        field = basis.interpolate(xvec)
        potential = asm(
            LinearForm(lambda v, w: 0.25 * (w.uh**2 - 1.0) ** 2 * v),
            basis,
            uh=field,
        ).sum()
        return 0.5 * epsilon**2 * xvec @ K @ xvec + potential

    e0 = energy(u)
    times, energies = [0.0], [e0]
    for step in range(1, steps + 1):
        un = basis.interpolate(u)
        A = M / dt + epsilon**2 * K + asm(cubic_linearization, basis, un=un)
        b = asm(right_hand_side, basis, un=un, dt=dt)
        u = spsolve(A, b)
        times.append(step * dt)
        energies.append(energy(u))
        if step % 50 == 0:
            print(f"step={step:3d}, time={step * dt:.3f}, energy={energies[-1]:.6e}")

    e1 = energy(u)
    print(f"energy: {e0:.6e} -> {e1:.6e}; range=[{u.min():.4f}, {u.max():.4f}]")
    assert np.isfinite(u).all() and e1 < e0

    triangulation = mtri.Triangulation(basis.mesh.p[0], basis.mesh.p[1], basis.mesh.t.T)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    limit = max(abs(initial_u).max(), abs(u).max())
    for ax, values, title in zip(axes[:2], (initial_u, u), ("Initial phase field", "Final phase field")):
        image = ax.tricontourf(triangulation, values, levels=21, cmap="coolwarm", vmin=-limit, vmax=limit)
        fig.colorbar(image, ax=ax, label="u")
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    axes[2].plot(times, energies)
    axes[2].set(title="Free-energy decay", xlabel="time", ylabel="energy")
    axes[2].grid(True, alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
