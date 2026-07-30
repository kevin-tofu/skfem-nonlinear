"""Fully implicit Allen--Cahn compared with a semi-implicit scheme."""

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


EPSILON = 0.04
TIME_STEP = 2.0e-3
STEPS = 250


@BilinearForm
def mass(u, v, w):
    return u * v


@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def weighted_mass(u, v, w):
    return w.weight * u * v


@LinearForm
def implicit_residual(v, w):
    return (
        (w.current - w.previous) * v / TIME_STEP
        + EPSILON**2 * dot(grad(w.current), grad(v))
        + (w.current**3 - w.current) * v
    )


@BilinearForm
def implicit_tangent(du, v, w):
    return (
        du * v / TIME_STEP
        + EPSILON**2 * dot(grad(du), grad(v))
        + (3.0 * w.current**2 - 1.0) * du * v
    )


@LinearForm
def semi_implicit_rhs(v, w):
    return (w.previous / TIME_STEP + w.previous) * v


@LinearForm
def potential_energy(v, w):
    return 0.25 * (w.phase**2 - 1.0) ** 2 * v


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 33), np.linspace(0.0, 1.0, 33))
    basis = Basis(mesh, ElementTriP1(), intorder=5)
    x, y = basis.doflocs
    initial = 0.35 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    mass_matrix = asm(mass, basis)
    stiffness_matrix = asm(stiffness, basis)

    def energy(state):
        potential = asm(
            potential_energy, basis, phase=basis.interpolate(state)
        ).sum()
        gradient = 0.5 * EPSILON**2 * state @ stiffness_matrix @ state
        return gradient + potential

    def step_residual(current, previous):
        return asm(
            implicit_residual,
            basis,
            current=basis.interpolate(current),
            previous=basis.interpolate(previous),
        )

    def step_jacobian(current):
        return asm(
            implicit_tangent,
            basis,
            current=basis.interpolate(current),
        )

    implicit = initial.copy()
    semi_implicit = initial.copy()
    times = [0.0]
    implicit_energies = [energy(implicit)]
    semi_implicit_energies = [energy(semi_implicit)]
    newton_iterations = []
    final_previous = None

    for step in range(1, STEPS + 1):
        previous = implicit.copy()
        current = previous.copy()
        initial_norm = None
        for iteration in range(1, 16):
            residual = step_residual(current, previous)
            norm_residual = np.linalg.norm(residual)
            initial_norm = norm_residual if initial_norm is None else initial_norm
            if norm_residual < max(1.0e-11, 1.0e-9 * initial_norm):
                break
            increment = spsolve(step_jacobian(current), -residual)
            alpha = 1.0
            while alpha > 1.0e-8:
                trial = current + alpha * increment
                if np.linalg.norm(step_residual(trial, previous)) < norm_residual:
                    current = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(f"line search failed at time step {step}")
        else:
            raise RuntimeError(f"Newton failed at time step {step}")
        implicit = current
        newton_iterations.append(iteration)
        final_previous = previous

        previous_semi = basis.interpolate(semi_implicit)
        semi_matrix = (
            mass_matrix / TIME_STEP
            + EPSILON**2 * stiffness_matrix
            + asm(weighted_mass, basis, weight=previous_semi**2)
        )
        semi_rhs = asm(semi_implicit_rhs, basis, previous=previous_semi)
        semi_implicit = spsolve(semi_matrix, semi_rhs)

        times.append(step * TIME_STEP)
        implicit_energies.append(energy(implicit))
        semi_implicit_energies.append(energy(semi_implicit))
        if step % 50 == 0:
            print(
                f"step={step:3d}, time={times[-1]:.3f}, Newton={iteration:2d}, "
                f"E_implicit={implicit_energies[-1]:.6e}, "
                f"E_semi={semi_implicit_energies[-1]:.6e}"
            )

    all_dofs = np.arange(basis.N)
    tangent_steps, tangent_errors = directional_derivative_errors(
        lambda state: step_residual(state, final_previous),
        step_jacobian,
        implicit,
        all_dofs,
    )
    relative_difference = np.linalg.norm(implicit - semi_implicit) / np.linalg.norm(implicit)
    print(
        f"relative scheme difference={relative_difference:.3e}, "
        f"minimum tangent error={tangent_errors.min():.3e}"
    )
    assert np.all(np.diff(implicit_energies) <= 1.0e-10)
    assert np.all(np.diff(semi_implicit_energies) <= 1.0e-10)
    assert tangent_errors.min() < 1.0e-7

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    limit = max(abs(implicit).max(), abs(semi_implicit).max())
    for ax, values, title in (
        (axes[0, 0], implicit, "Fully implicit phase field"),
        (axes[0, 1], semi_implicit, "Semi-implicit phase field"),
    ):
        image = ax.tricontourf(
            triangulation, values, levels=21, cmap="coolwarm", vmin=-limit, vmax=limit
        )
        fig.colorbar(image, ax=ax, label="phase")
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")

    difference_image = axes[0, 2].tricontourf(
        triangulation, implicit - semi_implicit, levels=21, cmap="PiYG"
    )
    fig.colorbar(difference_image, ax=axes[0, 2], label="implicit − semi-implicit")
    axes[0, 2].set(title="Difference at final time", xlabel="x", ylabel="y", aspect="equal")

    axes[1, 0].plot(times, implicit_energies, label="fully implicit")
    axes[1, 0].plot(times, semi_implicit_energies, "--", label="semi-implicit")
    axes[1, 0].set(title="Free-energy decay", xlabel="time", ylabel="energy")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].step(times[1:], newton_iterations, where="post")
    axes[1, 1].set(title="Fully implicit nonlinear work", xlabel="time", ylabel="Newton iterations")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].loglog(tangent_steps, tangent_errors, "o-", label="implicit tangent")
    axes[1, 2].loglog(
        tangent_steps,
        tangent_errors[0] * (tangent_steps / tangent_steps[0]) ** 2,
        "k--",
        label=r"$O(h^2)$",
    )
    axes[1, 2].invert_xaxis()
    axes[1, 2].set(title="Tangent verification", xlabel="difference step", ylabel="relative error")
    axes[1, 2].grid(True, which="both", alpha=0.3)
    axes[1, 2].legend()
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
