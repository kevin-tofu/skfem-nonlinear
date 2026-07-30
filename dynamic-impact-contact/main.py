"""Finite-strain elastic block impacting a rigid floor."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, FacetBasis, LinearForm
from skfem import MeshTri, asm
from skfem.helpers import ddot, dot, grad


MU, LAME_LAMBDA = 100.0, 200.0
DENSITY, GRAVITY = 1.0, 10.0
CONTACT_PENALTY = 5.0e4
TIME_STEP, FINAL_TIME = 5.0e-4, 0.6
NEWMARK_BETA, NEWMARK_GAMMA = 0.25, 0.5


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
def gravity_form(v, w):
    return -DENSITY * GRAVITY * v[1]


@LinearForm
def internal_force(v, w):
    return ddot(w.stress, grad(v))


@BilinearForm
def internal_tangent(du, v, w):
    return np.einsum("ijkl...,kl...,ij...->...", w.tangent, grad(du), grad(v))


@LinearForm
def contact_residual(v, w):
    return CONTACT_PENALTY * np.minimum(w.gap, 0.0) * v[1]


@BilinearForm
def contact_tangent(du, v, w):
    return CONTACT_PENALTY * (w.gap < 0.0) * du[1] * v[1]


@LinearForm
def integrate_scalar(v, w):
    return w.value * v


def main() -> None:
    mesh = MeshTri.init_tensor(
        np.linspace(-0.25, 0.25, 15),
        np.linspace(0.15, 0.65, 15),
    )
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=4)
    scalar_basis = Basis(mesh, ElementTriP1(), intorder=4)
    bottom_facets = mesh.facets_satisfying(lambda x: np.isclose(x[1], 0.15))
    contact_basis = FacetBasis(
        mesh, ElementVector(ElementTriP1()), facets=bottom_facets, intorder=4
    )
    contact_scalar_basis = FacetBasis(
        mesh, ElementTriP1(), facets=bottom_facets, intorder=4
    )
    mass_matrix = asm(mass_form, basis).tocsr()
    gravity_vector = asm(gravity_form, basis)
    identity = np.eye(2)[:, :, None, None]
    scalar_mass = asm(
        BilinearForm(lambda u, v, w: DENSITY * u * v), scalar_basis
    )
    nodal_mass = scalar_mass @ np.ones(scalar_basis.N)

    def state(displacement):
        F = identity + grad(basis.interpolate(displacement))
        return constitutive(F)

    def internal(displacement):
        _, stress, _, _ = state(displacement)
        return asm(internal_force, basis, stress=stress)

    def stiffness(displacement):
        _, _, tangent, _ = state(displacement)
        return asm(internal_tangent, basis, tangent=tangent)

    def contact_gap(displacement):
        field = contact_basis.interpolate(displacement)
        return contact_basis.global_coordinates()[1] + field[1]

    def contact_force(displacement):
        gap = contact_gap(displacement)
        return asm(contact_residual, contact_basis, gap=gap)

    def contact_stiffness(displacement):
        gap = contact_gap(displacement)
        return asm(contact_tangent, contact_basis, gap=gap)

    def strain_energy(displacement):
        density = state(displacement)[3]
        return asm(integrate_scalar, scalar_basis, value=density).sum()

    def contact_energy(displacement):
        gap = contact_gap(displacement)
        density = 0.5 * CONTACT_PENALTY * np.minimum(gap, 0.0) ** 2
        return asm(
            integrate_scalar, contact_scalar_basis, value=density
        ).sum()

    def total_contact_force(displacement):
        gap = contact_gap(displacement)
        traction = -CONTACT_PENALTY * np.minimum(gap, 0.0)
        return asm(
            integrate_scalar, contact_scalar_basis, value=traction
        ).sum()

    displacement = np.zeros(basis.N)
    velocity = np.zeros(basis.N)
    acceleration = spsolve(mass_matrix, gravity_vector)
    number_of_steps = int(round(FINAL_TIME / TIME_STEP))
    times = np.linspace(0.0, FINAL_TIME, number_of_steps + 1)
    centre_of_mass = [np.sum(nodal_mass * mesh.p[1]) / np.sum(nodal_mass)]
    vertical_velocity = [0.0]
    contact_forces = [0.0]
    minimum_gaps = [0.15]
    kinetic_energies = [0.0]
    strain_energies = [0.0]
    gravity_energies = [GRAVITY * np.sum(nodal_mass * mesh.p[1])]
    contact_energies = [0.0]
    iteration_history = [0]
    saved_states = [(0.0, displacement.copy())]

    for step in range(1, number_of_steps + 1):
        predicted_displacement = (
            displacement
            + TIME_STEP * velocity
            + TIME_STEP**2 * (0.5 - NEWMARK_BETA) * acceleration
        )
        predicted_velocity = velocity + TIME_STEP * (
            1.0 - NEWMARK_GAMMA
        ) * acceleration
        trial = predicted_displacement.copy()
        initial_norm = None
        for iteration in range(1, 21):
            trial_acceleration = (trial - predicted_displacement) / (
                NEWMARK_BETA * TIME_STEP**2
            )
            residual = (
                mass_matrix @ trial_acceleration
                + internal(trial)
                + contact_force(trial)
                - gravity_vector
            )
            norm_residual = np.linalg.norm(residual)
            initial_norm = norm_residual if initial_norm is None else initial_norm
            if norm_residual < max(1.0e-8, 1.0e-9 * initial_norm):
                break
            tangent = (
                mass_matrix / (NEWMARK_BETA * TIME_STEP**2)
                + stiffness(trial)
                + contact_stiffness(trial)
            )
            increment = spsolve(tangent, -residual)
            trial += increment
            if state(trial)[0].min() <= 0.0:
                raise RuntimeError(f"element inversion at time step {step}")
        else:
            raise RuntimeError(f"impact Newton failed at time step {step}")

        displacement = trial
        acceleration = (displacement - predicted_displacement) / (
            NEWMARK_BETA * TIME_STEP**2
        )
        velocity = predicted_velocity + NEWMARK_GAMMA * TIME_STEP * acceleration
        nodal_displacement = displacement[basis.nodal_dofs]
        nodal_velocity = velocity[basis.nodal_dofs]
        current_y = mesh.p[1] + nodal_displacement[1]
        centre_of_mass.append(np.sum(nodal_mass * current_y) / np.sum(nodal_mass))
        vertical_velocity.append(
            np.sum(nodal_mass * nodal_velocity[1]) / np.sum(nodal_mass)
        )
        contact_forces.append(total_contact_force(displacement))
        minimum_gaps.append(contact_gap(displacement).min())
        kinetic_energies.append(0.5 * velocity @ mass_matrix @ velocity)
        strain_energies.append(strain_energy(displacement))
        gravity_energies.append(GRAVITY * np.sum(nodal_mass * current_y))
        contact_energies.append(contact_energy(displacement))
        iteration_history.append(iteration)
        if step in (300, 350, 440, 600, 900, 1200):
            saved_states.append((times[step], displacement.copy()))
        if step % 300 == 0:
            print(
                f"step={step:3d}, time={times[step]:.3f}, "
                f"COM={centre_of_mass[-1]:.4f}, gap={minimum_gaps[-1]:+.2e}, "
                f"contact={contact_forces[-1]:.3f}, Newton={iteration}"
            )

    total_energy = (
        np.asarray(kinetic_energies)
        + np.asarray(strain_energies)
        + np.asarray(gravity_energies)
        + np.asarray(contact_energies)
    )
    energy_drift = (total_energy.max() - total_energy.min()) / total_energy[0]
    impact_indices = np.flatnonzero(np.asarray(contact_forces) > 1.0e-6)
    print(
        f"first impact time={times[impact_indices[0]]:.4f}, "
        f"peak contact force={max(contact_forces):.4f}, "
        f"minimum gap={min(minimum_gaps):.3e}, "
        f"relative energy drift={energy_drift:.3e}"
    )
    assert impact_indices.size > 0
    assert min(minimum_gaps) < 0.0
    assert energy_drift < 5.0e-3

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for time, saved_displacement in saved_states:
        nodal = saved_displacement[basis.nodal_dofs]
        deformed = mesh.p + nodal
        triangulation = mtri.Triangulation(
            deformed[0], deformed[1], mesh.t.T
        )
        axes[0, 0].triplot(
            triangulation,
            linewidth=0.45,
            label=f"t={time:g}",
        )
    axes[0, 0].axhline(0.0, color="black", linewidth=2, label="rigid floor")
    axes[0, 0].set(title="Impact and rebound configurations", xlabel="x", ylabel="y", aspect="equal")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(times, centre_of_mass)
    axes[0, 1].set(title="Centre-of-mass motion", xlabel="time", ylabel="vertical position")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(times, contact_forces)
    axes[0, 2].set(title="Impact force", xlabel="time", ylabel="floor contact force")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(times, minimum_gaps)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(title="Contact gap", xlabel="time", ylabel="minimum gap")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, kinetic_energies, label="kinetic")
    axes[1, 1].plot(times, strain_energies, label="strain")
    axes[1, 1].plot(times, gravity_energies, label="gravity")
    axes[1, 1].plot(times, contact_energies, label="contact")
    axes[1, 1].plot(times, total_energy, "k--", label="total")
    axes[1, 1].set(title="Energy exchange", xlabel="time", ylabel="energy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].step(times, iteration_history, where="post")
    axes[1, 2].set(title="Impact Newton work", xlabel="time", ylabel="iterations")
    axes[1, 2].grid(True, alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
