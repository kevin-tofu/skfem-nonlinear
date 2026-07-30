"""Fully implicit mixed formulation of the Cahn--Hilliard equation."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.nonlinear_verification import directional_derivative_errors


EPSILON = 0.035
TIME_STEP = 5.0e-4
STEPS = 300


@BilinearForm
def mass(u, v, w):
    return u * v


@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))


@LinearForm
def phase_residual(v, w):
    return (w.phase - w.previous) * v / TIME_STEP + dot(grad(w.chemical), grad(v))


@LinearForm
def chemical_residual(q, w):
    return (
        w.chemical * q
        - (w.phase**3 - w.phase) * q
        - EPSILON**2 * dot(grad(w.phase), grad(q))
    )


@BilinearForm
def chemical_phase_tangent(dc, q, w):
    return -(3.0 * w.phase**2 - 1.0) * dc * q - EPSILON**2 * dot(grad(dc), grad(q))


@LinearForm
def potential_energy(v, w):
    return 0.25 * (w.phase**2 - 1.0) ** 2 * v


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 33), np.linspace(0.0, 1.0, 33))
    basis = Basis(mesh, ElementTriP1(), intorder=5)
    n = basis.N
    M = asm(mass, basis)
    K = asm(stiffness, basis)
    generator = np.random.default_rng(7)
    phase = 0.04 * generator.standard_normal(n)
    phase -= np.ones(n) @ M @ phase / (np.ones(n) @ M @ np.ones(n))
    initial_phase = phase.copy()
    chemical = phase**3 - phase

    def split(state):
        return state[:n], state[n:]

    def residual(state, previous):
        current, potential = split(state)
        phase_field = basis.interpolate(current)
        chemical_field = basis.interpolate(potential)
        r_phase = asm(
            phase_residual,
            basis,
            phase=phase_field,
            previous=basis.interpolate(previous),
            chemical=chemical_field,
        )
        r_chemical = asm(
            chemical_residual,
            basis,
            phase=phase_field,
            chemical=chemical_field,
        )
        return np.concatenate((r_phase, r_chemical))

    def jacobian(state):
        current, _ = split(state)
        Kcc = M / TIME_STEP
        Kcm = K
        Kmc = asm(
            chemical_phase_tangent,
            basis,
            phase=basis.interpolate(current),
        )
        Kmm = M
        return bmat([[Kcc, Kcm], [Kmc, Kmm]], format="csr")

    def total_mass(state):
        return np.ones(n) @ M @ state

    def energy(state):
        potential = asm(
            potential_energy, basis, phase=basis.interpolate(state)
        ).sum()
        return 0.5 * EPSILON**2 * state @ K @ state + potential

    initial_mass = total_mass(phase)
    times = [0.0]
    energies = [energy(phase)]
    mass_errors = [0.0]
    newton_iterations = []
    final_previous = None

    for step in range(1, STEPS + 1):
        previous = phase.copy()
        state = np.concatenate((phase, chemical))
        initial_norm = None
        for iteration in range(1, 16):
            r = residual(state, previous)
            norm_r = np.linalg.norm(r)
            initial_norm = norm_r if initial_norm is None else initial_norm
            if norm_r < max(1.0e-10, 1.0e-9 * initial_norm):
                break
            increment = spsolve(jacobian(state), -r)
            alpha = 1.0
            while alpha > 1.0e-8:
                trial = state + alpha * increment
                if np.linalg.norm(residual(trial, previous)) < norm_r:
                    state = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(f"line search failed at step {step}")
        else:
            raise RuntimeError(f"Newton failed at step {step}")

        phase, chemical = split(state)
        final_previous = previous
        times.append(step * TIME_STEP)
        energies.append(energy(phase))
        mass_errors.append(abs(total_mass(phase) - initial_mass))
        newton_iterations.append(iteration)
        if step % 50 == 0:
            print(
                f"step={step:3d}, time={times[-1]:.4f}, Newton={iteration:2d}, "
                f"energy={energies[-1]:.6e}, mass error={mass_errors[-1]:.3e}, "
                f"phase=[{phase.min():+.3f}, {phase.max():+.3f}]"
            )

    final_state = np.concatenate((phase, chemical))
    all_dofs = np.arange(2 * n)
    tangent_steps, tangent_errors = directional_derivative_errors(
        lambda state: residual(state, final_previous),
        jacobian,
        final_state,
        all_dofs,
    )
    print(
        f"maximum mass error={max(mass_errors):.3e}, "
        f"minimum tangent error={tangent_errors.min():.3e}"
    )
    assert max(mass_errors) < 1.0e-10
    assert np.all(np.diff(energies) <= 1.0e-9)
    assert tangent_errors.min() < 1.0e-7

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    phase_limit = max(abs(initial_phase).max(), abs(phase).max())
    for ax, values, title in (
        (axes[0, 0], initial_phase, "Initial conserved phase"),
        (axes[0, 1], phase, "Phase-separated state"),
    ):
        image = ax.tricontourf(
            triangulation, values, levels=21, cmap="coolwarm", vmin=-phase_limit, vmax=phase_limit
        )
        fig.colorbar(image, ax=ax, label="phase")
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")

    chemical_image = axes[0, 2].tricontourf(
        triangulation, chemical, levels=21, cmap="PiYG"
    )
    fig.colorbar(chemical_image, ax=axes[0, 2], label="chemical potential")
    axes[0, 2].set(title="Chemical potential", xlabel="x", ylabel="y", aspect="equal")

    axes[1, 0].plot(times, energies)
    axes[1, 0].set(title="Free-energy decay", xlabel="time", ylabel="energy")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(times, np.maximum(mass_errors, np.finfo(float).eps), label="mass error")
    iteration_axis = axes[1, 1].twinx()
    iteration_axis.step(times[1:], newton_iterations, color="tab:orange", label="Newton iterations")
    axes[1, 1].set(title="Conservation and nonlinear work", xlabel="time", ylabel="absolute mass error")
    iteration_axis.set_ylabel("Newton iterations")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    lines = axes[1, 1].lines + iteration_axis.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], loc="best")

    axes[1, 2].loglog(tangent_steps, tangent_errors, "o-", label="mixed tangent")
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
