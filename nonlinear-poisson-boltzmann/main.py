"""Nonlinear Poisson--Boltzmann equation with charge continuation."""

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


KAPPA = 4.0
EXPONENTIAL_LIMIT = 20.0


def fixed_charge(x):
    positive = 180.0 * np.exp(-120.0 * ((x[0] - 0.32) ** 2 + (x[1] - 0.50) ** 2))
    negative = 180.0 * np.exp(-120.0 * ((x[0] - 0.68) ** 2 + (x[1] - 0.50) ** 2))
    return positive - negative


@LinearForm
def residual_form(v, w):
    safe_potential = np.clip(w.potential, -EXPONENTIAL_LIMIT, EXPONENTIAL_LIMIT)
    return (
        dot(grad(w.potential), grad(v))
        + KAPPA**2 * np.sinh(safe_potential) * v
        - w.load * w.charge * v
    )


@BilinearForm
def tangent_form(dpotential, v, w):
    safe_potential = np.clip(w.potential, -EXPONENTIAL_LIMIT, EXPONENTIAL_LIMIT)
    return dot(grad(dpotential), grad(v)) + KAPPA**2 * np.cosh(
        safe_potential
    ) * dpotential * v


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 49), np.linspace(0.0, 1.0, 49))
    basis = Basis(mesh, ElementTriP1(), intorder=6)
    fixed = basis.get_dofs().all()
    free = basis.complement_dofs(fixed)
    charge_qp = fixed_charge(basis.global_coordinates())

    def residual(state, load):
        return asm(
            residual_form,
            basis,
            potential=basis.interpolate(state),
            charge=charge_qp,
            load=load,
        )

    def jacobian(state):
        return asm(tangent_form, basis, potential=basis.interpolate(state))

    def newton(initial, load):
        state = initial.copy()
        history = []
        initial_norm = None
        for iteration in range(1, 31):
            r = residual(state, load)
            norm_r = np.linalg.norm(r[free])
            history.append(norm_r)
            initial_norm = norm_r if initial_norm is None else initial_norm
            if norm_r < max(1.0e-11, 1.0e-10 * initial_norm):
                break
            J = jacobian(state)
            increment = np.zeros_like(state)
            increment[free] = spsolve(J[free][:, free], -r[free])
            alpha = 1.0
            while alpha > 1.0e-9:
                trial = state + alpha * increment
                if (
                    np.max(np.abs(trial)) < EXPONENTIAL_LIMIT
                    and np.linalg.norm(residual(trial, load)[free]) < norm_r
                ):
                    state = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(f"line search failed at load={load}")
        else:
            raise RuntimeError(f"Newton failed at load={load}")
        return state, np.asarray(history)

    direct_solution, direct_history = newton(np.zeros(basis.N), 1.0)

    continuation_solution = np.zeros(basis.N)
    continuation_loads = np.linspace(0.1, 1.0, 10)
    continuation_iterations = []
    continuation_extrema = []
    for load in continuation_loads:
        continuation_solution, history = newton(continuation_solution, load)
        continuation_iterations.append(len(history))
        continuation_extrema.append(
            (continuation_solution.min(), continuation_solution.max())
        )
        print(
            f"load={load:.1f}, Newton={len(history):2d}, "
            f"phi=[{continuation_solution.min():+.6f}, "
            f"{continuation_solution.max():+.6f}], residual={history[-1]:.3e}"
        )

    relative_difference = np.linalg.norm(
        direct_solution - continuation_solution
    ) / np.linalg.norm(continuation_solution)
    tangent_steps, tangent_errors = directional_derivative_errors(
        lambda state: residual(state, 1.0),
        jacobian,
        continuation_solution,
        free,
    )
    print(
        f"direct Newton iterations={len(direct_history)}, "
        f"relative solution difference={relative_difference:.3e}, "
        f"minimum tangent error={tangent_errors.min():.3e}"
    )
    assert relative_difference < 1.0e-8
    assert tangent_errors.min() < 1.0e-7

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    ionic_charge = KAPPA**2 * np.sinh(continuation_solution)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    potential_image = axes[0, 0].tricontourf(
        triangulation, continuation_solution, levels=25, cmap="coolwarm"
    )
    fig.colorbar(potential_image, ax=axes[0, 0], label="electrostatic potential")
    axes[0, 0].set(title="Poisson–Boltzmann potential", xlabel="x", ylabel="y", aspect="equal")

    charge_image = axes[0, 1].tricontourf(
        triangulation, ionic_charge, levels=25, cmap="PuOr"
    )
    fig.colorbar(charge_image, ax=axes[0, 1], label=r"$\kappa^2\sinh(\phi)$")
    axes[0, 1].set(title="Mobile ionic charge", xlabel="x", ylabel="y", aspect="equal")

    extrema = np.asarray(continuation_extrema)
    axes[1, 0].plot(continuation_loads, extrema[:, 1], "o-", label=r"$\max\phi$")
    axes[1, 0].plot(continuation_loads, extrema[:, 0], "o-", label=r"$\min\phi$")
    iteration_axis = axes[1, 0].twinx()
    iteration_axis.step(
        continuation_loads,
        continuation_iterations,
        where="mid",
        color="tab:green",
        label="Newton iterations",
    )
    axes[1, 0].set(title="Charge continuation", xlabel="charge scale", ylabel="potential extrema")
    iteration_axis.set_ylabel("Newton iterations")
    axes[1, 0].grid(True, alpha=0.3)
    lines = axes[1, 0].lines + iteration_axis.lines
    axes[1, 0].legend(lines, [line.get_label() for line in lines], loc="best")

    axes[1, 1].loglog(tangent_steps, tangent_errors, "o-", label="PB tangent")
    axes[1, 1].loglog(
        tangent_steps,
        tangent_errors[0] * (tangent_steps / tangent_steps[0]) ** 2,
        "k--",
        label=r"$O(h^2)$",
    )
    axes[1, 1].invert_xaxis()
    axes[1, 1].set(title="Tangent verification", xlabel="difference step", ylabel="relative error")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
