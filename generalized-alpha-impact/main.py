"""Newmark and generalized-alpha integration of a nonlinear bar impact."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementLineP1, LinearForm, MeshLine, asm
from skfem.helpers import dot, grad


YOUNG, DENSITY, GRAVITY = 100.0, 1.0, 10.0
CONTACT_PENALTY = 5.0e4
TIME_STEP, FINAL_TIME = 5.0e-4, 0.45


@BilinearForm
def mass_form(u, v, w):
    return DENSITY * u * v


@LinearForm
def gravity_form(v, w):
    return -DENSITY * GRAVITY * v


@LinearForm
def internal_force(v, w):
    deformation_gradient = 1.0 + grad(w.displacement)[0]
    first_piola = 0.5 * YOUNG * (
        deformation_gradient - 1.0 / deformation_gradient
    )
    return first_piola * grad(v)[0]


@BilinearForm
def internal_tangent(du, v, w):
    deformation_gradient = 1.0 + grad(w.displacement)[0]
    material_tangent = 0.5 * YOUNG * (
        1.0 + 1.0 / deformation_gradient**2
    )
    return material_tangent * grad(du)[0] * grad(v)[0]


@LinearForm
def strain_energy_form(v, w):
    deformation_gradient = 1.0 + grad(w.displacement)[0]
    density = (
        0.25 * YOUNG * (deformation_gradient**2 - 1.0)
        - 0.5 * YOUNG * np.log(deformation_gradient)
    )
    return density * v


def parameters(rho_infinity):
    alpha_m = (2.0 * rho_infinity - 1.0) / (rho_infinity + 1.0)
    alpha_f = rho_infinity / (rho_infinity + 1.0)
    gamma = 0.5 + alpha_f - alpha_m
    beta = 0.25 * (1.0 + alpha_f - alpha_m) ** 2
    return alpha_m, alpha_f, beta, gamma


def main() -> None:
    mesh = MeshLine(np.linspace(0.15, 0.65, 81)[None, :])
    basis = Basis(mesh, ElementLineP1(), intorder=5)
    mass_matrix = asm(mass_form, basis).tocsr()
    gravity_vector = asm(gravity_form, basis)
    bottom = int(np.argmin(basis.doflocs[0]))
    nodal_mass = mass_matrix @ np.ones(basis.N)

    def force(displacement):
        return asm(
            internal_force,
            basis,
            displacement=basis.interpolate(displacement),
        )

    def stiffness(displacement):
        return asm(
            internal_tangent,
            basis,
            displacement=basis.interpolate(displacement),
        )

    def strain_energy(displacement):
        return asm(
            strain_energy_form,
            basis,
            displacement=basis.interpolate(displacement),
        ).sum()

    def gap(displacement):
        return basis.doflocs[0, bottom] + displacement[bottom]

    def contact_residual(displacement):
        result = np.zeros(basis.N)
        result[bottom] = CONTACT_PENALTY * min(gap(displacement), 0.0)
        return result

    def contact_stiffness(displacement):
        diagonal = np.zeros(basis.N)
        diagonal[bottom] = CONTACT_PENALTY if gap(displacement) < 0.0 else 0.0
        return diags(diagonal)

    times = np.arange(0.0, FINAL_TIME + 0.5 * TIME_STEP, TIME_STEP)

    def integrate(label, rho_infinity=None):
        if rho_infinity is None:
            alpha_m, alpha_f, beta, gamma = 0.0, 0.0, 0.25, 0.5
        else:
            alpha_m, alpha_f, beta, gamma = parameters(rho_infinity)
        displacement = np.zeros(basis.N)
        velocity = np.zeros(basis.N)
        acceleration = spsolve(mass_matrix, gravity_vector)
        centre = [np.sum(nodal_mass * basis.doflocs[0]) / np.sum(nodal_mass)]
        centre_velocity = [0.0]
        contact_force = [0.0]
        total_energy = [
            GRAVITY * np.sum(nodal_mass * basis.doflocs[0])
        ]
        iterations = [0]

        for step in range(1, times.size):
            old_displacement = displacement.copy()
            old_acceleration = acceleration.copy()
            predicted_displacement = (
                displacement
                + TIME_STEP * velocity
                + TIME_STEP**2 * (0.5 - beta) * acceleration
            )
            predicted_velocity = velocity + TIME_STEP * (1.0 - gamma) * acceleration
            trial = predicted_displacement.copy()
            for iteration in range(1, 16):
                trial_acceleration = (trial - predicted_displacement) / (
                    beta * TIME_STEP**2
                )
                evaluated_displacement = (
                    (1.0 - alpha_f) * trial
                    + alpha_f * old_displacement
                )
                evaluated_acceleration = (
                    (1.0 - alpha_m) * trial_acceleration
                    + alpha_m * old_acceleration
                )
                residual = (
                    mass_matrix @ evaluated_acceleration
                    + force(evaluated_displacement)
                    + contact_residual(evaluated_displacement)
                    - gravity_vector
                )
                if np.linalg.norm(residual) < 1.0e-8:
                    break
                effective_tangent = (
                    (1.0 - alpha_m)
                    * mass_matrix
                    / (beta * TIME_STEP**2)
                    + (1.0 - alpha_f)
                    * (
                        stiffness(evaluated_displacement)
                        + contact_stiffness(evaluated_displacement)
                    )
                )
                trial += spsolve(effective_tangent, -residual)
            else:
                raise RuntimeError(f"{label} failed at step {step}")

            displacement = trial
            acceleration = (displacement - predicted_displacement) / (
                beta * TIME_STEP**2
            )
            velocity = predicted_velocity + gamma * TIME_STEP * acceleration
            current_position = basis.doflocs[0] + displacement
            centre.append(np.sum(nodal_mass * current_position) / np.sum(nodal_mass))
            centre_velocity.append(
                np.sum(nodal_mass * velocity) / np.sum(nodal_mass)
            )
            contact_force.append(
                -CONTACT_PENALTY * min(gap(displacement), 0.0)
            )
            kinetic = 0.5 * velocity @ mass_matrix @ velocity
            gravity_energy = GRAVITY * np.sum(nodal_mass * current_position)
            contact_energy = 0.5 * CONTACT_PENALTY * min(
                gap(displacement), 0.0
            ) ** 2
            total_energy.append(
                kinetic
                + strain_energy(displacement)
                + gravity_energy
                + contact_energy
            )
            iterations.append(iteration)
        return {
            "label": label,
            "centre": np.asarray(centre),
            "velocity": np.asarray(centre_velocity),
            "force": np.asarray(contact_force),
            "energy": np.asarray(total_energy),
            "iterations": np.asarray(iterations),
        }

    results = [
        integrate("Newmark"),
        integrate(r"$\rho_\infty=0.5$", 0.5),
        integrate(r"$\rho_\infty=0$", 0.0),
    ]
    impact_mask = times > 0.15
    frequencies = np.fft.rfftfreq(np.count_nonzero(impact_mask), TIME_STEP)
    high_frequency = frequencies > 200.0
    spectral_measures = []
    for result in results:
        spectrum = np.abs(
            np.fft.rfft(result["force"][impact_mask])
        )
        spectral_measures.append(np.linalg.norm(spectrum[high_frequency]))
        print(
            f"{result['label']}: peak force={result['force'].max():.4f}, "
            f"energy loss={(result['energy'][0] - result['energy'][-1]) / result['energy'][0]:.3e}, "
            f"max Newton={result['iterations'].max()}"
        )
    assert spectral_measures[2] < spectral_measures[0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for result in results:
        axes[0, 0].plot(times, result["centre"], label=result["label"])
        axes[0, 1].plot(times, result["force"], label=result["label"])
        axes[0, 2].plot(
            result["centre"], result["velocity"], label=result["label"]
        )
        axes[1, 0].plot(
            times,
            result["energy"] / result["energy"][0],
            label=result["label"],
        )
        spectrum = np.abs(np.fft.rfft(result["force"][impact_mask]))
        axes[1, 1].semilogy(
            frequencies[1:], spectrum[1:], label=result["label"]
        )
        axes[1, 2].step(
            times,
            result["iterations"],
            where="post",
            label=result["label"],
        )
    axes[0, 0].set(title="Impact and rebound", xlabel="time", ylabel="centre position")
    axes[0, 1].set(title="Contact-force filtering", xlabel="time", ylabel="contact force")
    axes[0, 2].set(title="Phase trajectory", xlabel="centre position", ylabel="centre velocity")
    axes[1, 0].set(title="Algorithmic energy decay", xlabel="time", ylabel="normalized total energy")
    axes[1, 1].set(title="Contact-force spectrum", xlabel="frequency", ylabel="amplitude", xlim=(1.0, 1000.0))
    axes[1, 1].set_xscale("log")
    axes[1, 2].set(title="Newton work", xlabel="time", ylabel="iterations")
    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
