"""Regularized p-Laplacian for several values of p on the unit square."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.nonlinear_verification import directional_derivative_errors


REGULARIZATION = 1.0e-3
SOURCE = 1.0


@LinearForm
def residual_form(v, w):
    gradient = grad(w.uh)
    squared_norm = dot(gradient, gradient)
    coefficient = (REGULARIZATION**2 + squared_norm) ** ((w.p - 2.0) / 2.0)
    return coefficient * dot(gradient, grad(v)) - SOURCE * v


@BilinearForm
def tangent_form(du, v, w):
    gradient = grad(w.uh)
    squared_norm = dot(gradient, gradient)
    base = REGULARIZATION**2 + squared_norm
    isotropic = base ** ((w.p - 2.0) / 2.0) * dot(grad(du), grad(v))
    directional = (
        (w.p - 2.0)
        * base ** ((w.p - 4.0) / 2.0)
        * dot(gradient, grad(du))
        * dot(gradient, grad(v))
    )
    return isotropic + directional


def solve_problem(basis, free, p, initial):
    def residual(state):
        return asm(residual_form, basis, uh=basis.interpolate(state), p=p)

    def jacobian(state):
        return asm(tangent_form, basis, uh=basis.interpolate(state), p=p)

    u = initial.copy()
    history = []
    initial_norm = None
    for iteration in range(1, 41):
        r = residual(u)
        norm_r = np.linalg.norm(r[free])
        history.append(norm_r)
        initial_norm = norm_r if initial_norm is None else initial_norm
        if norm_r < max(1.0e-11, 1.0e-9 * initial_norm):
            break
        J = jacobian(u)
        increment = np.zeros_like(u)
        increment[free] = spsolve(J[free][:, free], -r[free])
        alpha = 1.0
        while alpha > 1.0e-9:
            trial = u + alpha * increment
            if np.linalg.norm(residual(trial)[free]) < norm_r:
                u = trial
                break
            alpha *= 0.5
        else:
            raise RuntimeError(f"line search failed for p={p}")
    else:
        raise RuntimeError(f"Newton failed for p={p}")
    print(
        f"p={p:3.1f}: Newton={iteration:2d}, max(u)={u.max():.6f}, "
        f"residual={history[-1]:.3e}"
    )
    return u, np.asarray(history), residual, jacobian


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 41), np.linspace(0.0, 1.0, 41))
    basis = Basis(mesh, ElementTriP1(), intorder=5)
    fixed = basis.get_dofs().all()
    free = basis.complement_dofs(fixed)

    # The p=2 solution is a useful initial guess on both sides of p=2.
    linear_solution, linear_history, _, _ = solve_problem(
        basis, free, 2.0, np.zeros(basis.N)
    )
    solutions = {2.0: linear_solution}
    histories = {2.0: linear_history}
    checks = {}
    for p in (1.5, 3.0, 4.0):
        solution, history, residual, jacobian = solve_problem(
            basis, free, p, linear_solution
        )
        solutions[p] = solution
        histories[p] = history
        if p in (1.5, 3.0):
            checks[p] = directional_derivative_errors(
                residual, jacobian, solution, free
            )

    assert all(np.isfinite(solution).all() for solution in solutions.values())
    assert all(errors.min() < 1.0e-7 for _, errors in checks.values())

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    image = axes[0, 0].tricontourf(
        triangulation, solutions[1.5], levels=24, cmap="viridis"
    )
    fig.colorbar(image, ax=axes[0, 0], label="u")
    axes[0, 0].set(title=r"Regularized solution, $p=1.5$", xlabel="x", ylabel="y", aspect="equal")

    centre = np.isclose(basis.doflocs[1], 0.5)
    order = np.argsort(basis.doflocs[0, centre])
    for p in sorted(solutions):
        axes[0, 1].plot(
            basis.doflocs[0, centre][order],
            solutions[p][centre][order],
            label=f"p={p:g}",
        )
    axes[0, 1].set(title="Horizontal centre-line profiles", xlabel="x", ylabel="u")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    p_values = sorted(solutions)
    axes[1, 0].plot(p_values, [solutions[p].max() for p in p_values], "o-")
    axes[1, 0].set(title="Effect of the exponent", xlabel="p", ylabel=r"$\max u$")
    axes[1, 0].grid(True, alpha=0.3)

    for p, (steps, errors) in checks.items():
        axes[1, 1].loglog(steps, errors, "o-", label=f"p={p:g}")
    reference_steps = checks[1.5][0]
    reference_error = checks[1.5][1][0]
    axes[1, 1].loglog(
        reference_steps,
        reference_error * (reference_steps / reference_steps[0]) ** 2,
        "k--",
        label=r"$O(h^2)$",
    )
    axes[1, 1].invert_xaxis()
    axes[1, 1].set(title="Consistent-tangent verification", xlabel="difference step", ylabel="relative error")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
