"""Finite-strain cantilever dynamics using implicit Newmark and Newton."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, FacetBasis, LinearForm
from skfem import MeshTri, asm
from skfem.helpers import ddot, dot, grad


MU, LAME_LAMBDA = 100.0, 200.0
DENSITY = 1.0
LENGTH, HEIGHT = 2.0, 0.4
TIME_STEP, FINAL_TIME = 0.01, 8.0
NEWMARK_BETA, NEWMARK_GAMMA = 0.25, 0.5
PULSE_DURATION, PEAK_LOAD = 0.5, 0.20


def constitutive(F):
    moved = np.moveaxis(F, (0, 1), (-2, -1))
    determinant = np.linalg.det(moved)
    inverse_transpose = np.swapaxes(np.linalg.inv(moved), -1, -2)
    inverse_transpose = np.moveaxis(inverse_transpose, (-2, -1), (0, 1))
    logarithm = np.log(determinant)
    stress = (
        MU * (F - inverse_transpose)
        + LAME_LAMBDA * logarithm[None, None, ...] * inverse_transpose
    )
    tangent = np.zeros((2, 2, 2, 2) + determinant.shape)
    identity = np.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for ell in range(2):
                    tangent[i, j, k, ell] = (
                        MU * identity[i, k] * identity[j, ell]
                        + (MU - LAME_LAMBDA * logarithm)
                        * inverse_transpose[i, ell]
                        * inverse_transpose[k, j]
                        + LAME_LAMBDA
                        * inverse_transpose[i, j]
                        * inverse_transpose[k, ell]
                    )
    energy = (
        0.5 * MU * (np.einsum("ij...,ij...->...", F, F) - 2.0)
        - MU * logarithm
        + 0.5 * LAME_LAMBDA * logarithm**2
    )
    return determinant, stress, tangent, energy


@BilinearForm
def mass_form(u, v, w):
    return DENSITY * dot(u, v)


@LinearForm
def internal_force(v, w):
    return ddot(w.stress, grad(v))


@BilinearForm
def internal_tangent(du, v, w):
    return np.einsum("ijkl...,kl...,ij...->...", w.tangent, grad(du), grad(v))


@LinearForm
def pulse_load(v, w):
    return -v[1] / HEIGHT


@LinearForm
def integrate_energy(v, w):
    return w.energy * v


def main() -> None:
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, LENGTH, 41),
        np.linspace(-HEIGHT / 2.0, HEIGHT / 2.0, 9),
    )
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=4)
    scalar_basis = Basis(mesh, ElementTriP1(), intorder=4)
    right_facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], LENGTH))
    right_basis = FacetBasis(
        mesh, ElementVector(ElementTriP1()), facets=right_facets, intorder=4
    )
    fixed = basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    free = basis.complement_dofs(fixed)
    right = basis.get_dofs(lambda x: np.isclose(x[0], LENGTH))
    right_y = right.nodal["u^2"]
    mass_matrix = asm(mass_form, basis).tocsr()
    reference_load = asm(pulse_load, right_basis)
    identity = np.eye(2)[:, :, None, None]

    def state(displacement):
        F = identity + grad(basis.interpolate(displacement))
        return constitutive(F)

    def force(displacement):
        _, stress, _, _ = state(displacement)
        return asm(internal_force, basis, stress=stress)

    def tangent(displacement):
        _, _, material, _ = state(displacement)
        return asm(internal_tangent, basis, tangent=material)

    def strain_energy(displacement):
        energy_density = state(displacement)[3]
        return asm(
            integrate_energy, scalar_basis, energy=energy_density
        ).sum()

    def load_amplitude(time):
        if time <= PULSE_DURATION:
            return PEAK_LOAD * np.sin(np.pi * time / PULSE_DURATION)
        return 0.0

    number_of_steps = int(round(FINAL_TIME / TIME_STEP))
    times = np.linspace(0.0, FINAL_TIME, number_of_steps + 1)
    nonlinear_u = np.zeros(basis.N)
    nonlinear_v = np.zeros(basis.N)
    nonlinear_a = np.zeros(basis.N)
    linear_u = np.zeros(basis.N)
    linear_v = np.zeros(basis.N)
    linear_a = np.zeros(basis.N)
    initial_stiffness = tangent(np.zeros(basis.N)).tocsr()

    nonlinear_tip = [0.0]
    nonlinear_tip_velocity = [0.0]
    linear_tip = [0.0]
    kinetic_history = [0.0]
    strain_history = [0.0]
    work_history = [0.0]
    iteration_history = [0]
    saved_states = [(0.0, nonlinear_u.copy())]
    accumulated_work = 0.0

    effective_linear = (
        mass_matrix / (NEWMARK_BETA * TIME_STEP**2) + initial_stiffness
    )
    effective_linear_free = effective_linear[free][:, free]

    for step in range(1, number_of_steps + 1):
        time = times[step]
        amplitude = load_amplitude(time)
        external = amplitude * reference_load

        predicted_u = (
            nonlinear_u
            + TIME_STEP * nonlinear_v
            + TIME_STEP**2 * (0.5 - NEWMARK_BETA) * nonlinear_a
        )
        predicted_v = nonlinear_v + TIME_STEP * (
            1.0 - NEWMARK_GAMMA
        ) * nonlinear_a
        trial_u = predicted_u.copy()
        initial_norm = None
        for iteration in range(1, 16):
            trial_a = (trial_u - predicted_u) / (
                NEWMARK_BETA * TIME_STEP**2
            )
            residual = mass_matrix @ trial_a + force(trial_u) - external
            norm_residual = np.linalg.norm(residual[free])
            initial_norm = norm_residual if initial_norm is None else initial_norm
            if norm_residual < max(1.0e-9, 1.0e-9 * initial_norm):
                break
            effective_tangent = (
                mass_matrix / (NEWMARK_BETA * TIME_STEP**2)
                + tangent(trial_u)
            )
            increment = np.zeros_like(trial_u)
            increment[free] = spsolve(
                effective_tangent[free][:, free], -residual[free]
            )
            trial_u += increment
        else:
            raise RuntimeError(f"dynamic Newton failed at step {step}")

        old_u = nonlinear_u.copy()
        nonlinear_u = trial_u
        nonlinear_a = (nonlinear_u - predicted_u) / (
            NEWMARK_BETA * TIME_STEP**2
        )
        nonlinear_v = predicted_v + (
            NEWMARK_GAMMA * TIME_STEP * nonlinear_a
        )

        linear_predicted_u = (
            linear_u
            + TIME_STEP * linear_v
            + TIME_STEP**2 * (0.5 - NEWMARK_BETA) * linear_a
        )
        linear_predicted_v = linear_v + TIME_STEP * (
            1.0 - NEWMARK_GAMMA
        ) * linear_a
        linear_rhs = (
            external
            + mass_matrix
            @ (linear_predicted_u / (NEWMARK_BETA * TIME_STEP**2))
        )
        linear_u_new = np.zeros_like(linear_u)
        linear_u_new[free] = spsolve(
            effective_linear_free, linear_rhs[free]
        )
        linear_u = linear_u_new
        linear_a = (linear_u - linear_predicted_u) / (
            NEWMARK_BETA * TIME_STEP**2
        )
        linear_v = linear_predicted_v + NEWMARK_GAMMA * TIME_STEP * linear_a

        previous_amplitude = load_amplitude(times[step - 1])
        average_external = 0.5 * (
            previous_amplitude + amplitude
        ) * reference_load
        accumulated_work += average_external @ (nonlinear_u - old_u)
        kinetic = 0.5 * nonlinear_v @ mass_matrix @ nonlinear_v
        strain = strain_energy(nonlinear_u)
        nonlinear_tip.append(np.mean(nonlinear_u[right_y]))
        nonlinear_tip_velocity.append(np.mean(nonlinear_v[right_y]))
        linear_tip.append(np.mean(linear_u[right_y]))
        kinetic_history.append(kinetic)
        strain_history.append(strain)
        work_history.append(accumulated_work)
        iteration_history.append(iteration)
        if step in (50, 200, 400, 800):
            saved_states.append((time, nonlinear_u.copy()))
        if step % 200 == 0:
            print(
                f"step={step:3d}, time={time:.2f}, tip={nonlinear_tip[-1]:+.5f}, "
                f"energy={kinetic + strain:.6e}, Newton={iteration}"
            )

    total_energy = np.asarray(kinetic_history) + np.asarray(strain_history)
    after_pulse = times >= PULSE_DURATION
    energy_drift = (
        total_energy[after_pulse].max() - total_energy[after_pulse].min()
    ) / total_energy[after_pulse].mean()
    response_difference = np.linalg.norm(
        np.asarray(nonlinear_tip) - np.asarray(linear_tip)
    ) / np.linalg.norm(nonlinear_tip)
    print(
        f"relative nonlinear/linear difference={response_difference:.3e}, "
        f"post-pulse energy drift={energy_drift:.3e}, "
        f"maximum Newton iterations={max(iteration_history)}"
    )
    assert response_difference > 1.0e-3
    assert energy_drift < 5.0e-3

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for time, displacement in saved_states:
        nodal = displacement[basis.nodal_dofs]
        deformed = mesh.p + nodal
        triangulation = mtri.Triangulation(
            deformed[0], deformed[1], mesh.t.T
        )
        axes[0, 0].triplot(
            triangulation,
            linewidth=0.45,
            label=f"t={time:g}",
        )
    axes[0, 0].set(title="Dynamic configurations", xlabel="x", ylabel="y", aspect="equal")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(times, nonlinear_tip, label="finite strain")
    axes[0, 1].plot(times, linear_tip, "--", label="linear reference")
    axes[0, 1].axvspan(0.0, PULSE_DURATION, color="tab:orange", alpha=0.15, label="pulse")
    axes[0, 1].set(title="Tip vibration", xlabel="time", ylabel="vertical displacement")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].plot(nonlinear_tip, nonlinear_tip_velocity)
    axes[0, 2].set(title="Nonlinear phase trajectory", xlabel="tip displacement", ylabel="tip velocity")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(times, kinetic_history, label="kinetic")
    axes[1, 0].plot(times, strain_history, label="strain")
    axes[1, 0].plot(times, total_energy, "k--", label="total")
    axes[1, 0].set(title="Mechanical energy", xlabel="time", ylabel="energy")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(times, total_energy - np.asarray(work_history))
    axes[1, 1].set(title="Energy-balance error", xlabel="time", ylabel="energy − external work")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].step(times, iteration_history, where="post")
    axes[1, 2].set(title="Implicit dynamic Newton work", xlabel="time", ylabel="iterations")
    axes[1, 2].grid(True, alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
