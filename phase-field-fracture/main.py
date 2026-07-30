"""AT2 phase-field fracture in antiplane shear using alternate minimization."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm, condense, solve
from skfem.helpers import dot, grad


SHEAR_MODULUS = 1.0
FRACTURE_TOUGHNESS = 8.0e-3
LENGTH_SCALE = 0.025
RESIDUAL_STIFFNESS = 1.0e-6


@BilinearForm
def degraded_elasticity(u, v, w):
    degradation = (1.0 - w.damage) ** 2 + RESIDUAL_STIFFNESS
    return SHEAR_MODULUS * degradation * dot(grad(u), grad(v))


@BilinearForm
def damage_operator(d, q, w):
    return (
        FRACTURE_TOUGHNESS * LENGTH_SCALE * dot(grad(d), grad(q))
        + (FRACTURE_TOUGHNESS / LENGTH_SCALE + 2.0 * w.history) * d * q
    )


@LinearForm
def damage_source(q, w):
    return 2.0 * w.history * q


@BilinearForm
def mass(u, v, w):
    return u * v


@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 65), np.linspace(0.0, 1.0, 33))
    basis = Basis(mesh, ElementTriP1(), intorder=4)
    bottom = basis.get_dofs(lambda x: np.isclose(x[1], 0.0)).all()
    top = basis.get_dofs(lambda x: np.isclose(x[1], 1.0)).all()
    displacement_fixed = np.unique(np.concatenate((bottom, top)))
    precrack = basis.get_dofs(
        lambda x: np.isclose(x[1], 0.5) & (x[0] <= 0.18)
    ).all()

    displacement = np.zeros(basis.N)
    damage = np.zeros(basis.N)
    damage[precrack] = 1.0
    initial_damage = damage.copy()
    history = np.zeros(basis.global_coordinates().shape[1:])
    M = asm(mass, basis)
    K = asm(stiffness, basis)

    loads = np.linspace(0.02, 0.65, 32)
    reactions = []
    crack_lengths = []
    elastic_energies = []
    fracture_energies = []
    staggered_iterations = []

    for load_step, imposed in enumerate(loads, 1):
        damage_previous_step = damage.copy()
        history_previous_step = history.copy()
        for iteration in range(1, 251):
            old_displacement = displacement.copy()
            old_damage = damage.copy()

            elastic_matrix = asm(
                degraded_elasticity,
                basis,
                damage=basis.interpolate(damage),
            )
            prescribed_u = np.zeros(basis.N)
            prescribed_u[top] = imposed
            displacement = solve(
                *condense(
                    elastic_matrix,
                    np.zeros(basis.N),
                    x=prescribed_u,
                    D=displacement_fixed,
                )
            )

            gradient_u = grad(basis.interpolate(displacement))
            current_energy_density = 0.5 * SHEAR_MODULUS * dot(
                gradient_u, gradient_u
            )
            trial_history = np.maximum(history_previous_step, current_energy_density)
            damage_matrix = asm(
                damage_operator,
                basis,
                history=trial_history,
            )
            damage_rhs = asm(damage_source, basis, history=trial_history)
            prescribed_damage = np.zeros(basis.N)
            prescribed_damage[precrack] = 1.0
            candidate_damage = solve(
                *condense(
                    damage_matrix,
                    damage_rhs,
                    x=prescribed_damage,
                    D=precrack,
                )
            )
            damage = np.clip(
                np.maximum(candidate_damage, damage_previous_step),
                0.0,
                1.0,
            )
            relative_change = max(
                np.linalg.norm(displacement - old_displacement)
                / max(np.linalg.norm(displacement), 1.0),
                np.linalg.norm(damage - old_damage)
                / max(np.linalg.norm(damage), 1.0),
            )
            if relative_change < 1.0e-6:
                history = trial_history
                break
        else:
            raise RuntimeError(f"alternate minimization failed at step {load_step}")

        elastic_matrix = asm(
            degraded_elasticity,
            basis,
            damage=basis.interpolate(damage),
        )
        reaction = (elastic_matrix @ displacement)[top].sum()
        elastic_energy = 0.5 * displacement @ elastic_matrix @ displacement
        fracture_energy = FRACTURE_TOUGHNESS * (
            0.5 / LENGTH_SCALE * damage @ M @ damage
            + 0.5 * LENGTH_SCALE * damage @ K @ damage
        )
        centre_band = np.abs(basis.doflocs[1] - 0.5) < 1.5 / 32.0
        cracked = centre_band & (damage > 0.9)
        crack_length = (
            basis.doflocs[0, cracked].max() if np.any(cracked) else 0.0
        )
        reactions.append(reaction)
        crack_lengths.append(crack_length)
        elastic_energies.append(elastic_energy)
        fracture_energies.append(fracture_energy)
        staggered_iterations.append(iteration)
        print(
            f"step={load_step:2d}, imposed={imposed:.3f}, reaction={reaction:.4f}, "
            f"crack={crack_length:.3f}, max(d)={damage.max():.3f}, "
            f"staggered={iteration:2d}"
        )

    assert np.all(np.diff(crack_lengths) >= -1.0e-12)
    assert crack_lengths[-1] > crack_lengths[0] + 0.1
    assert np.all(damage >= initial_damage - 1.0e-12)

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    initial_image = axes[0, 0].tricontourf(
        triangulation, initial_damage, levels=np.linspace(0.0, 1.0, 21), cmap="inferno"
    )
    fig.colorbar(initial_image, ax=axes[0, 0], label="damage")
    axes[0, 0].set(title="Initial notch", xlabel="x", ylabel="y", aspect="equal")

    damage_image = axes[0, 1].tricontourf(
        triangulation, damage, levels=np.linspace(0.0, 1.0, 21), cmap="inferno"
    )
    fig.colorbar(damage_image, ax=axes[0, 1], label="damage")
    axes[0, 1].set(title="Propagated phase-field crack", xlabel="x", ylabel="y", aspect="equal")

    displacement_image = axes[0, 2].tricontourf(
        triangulation, displacement, levels=24, cmap="coolwarm"
    )
    fig.colorbar(displacement_image, ax=axes[0, 2], label="antiplane displacement")
    axes[0, 2].set(title="Final displacement field", xlabel="x", ylabel="y", aspect="equal")

    axes[1, 0].plot(loads, reactions, "o-", markersize=3)
    axes[1, 0].set(title="Load–displacement response", xlabel="imposed displacement", ylabel="reaction")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(loads, elastic_energies, label="elastic")
    axes[1, 1].plot(loads, fracture_energies, label="fracture")
    axes[1, 1].plot(
        loads,
        np.asarray(elastic_energies) + np.asarray(fracture_energies),
        "--",
        label="total internal",
    )
    axes[1, 1].set(title="Energy evolution", xlabel="imposed displacement", ylabel="energy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].plot(loads, crack_lengths, "o-", label="crack length")
    iteration_axis = axes[1, 2].twinx()
    iteration_axis.step(
        loads,
        staggered_iterations,
        where="mid",
        color="tab:orange",
        label="staggered iterations",
    )
    axes[1, 2].set(title="Irreversible crack growth", xlabel="imposed displacement", ylabel="crack length")
    iteration_axis.set_ylabel("alternate iterations")
    axes[1, 2].grid(True, alpha=0.3)
    lines = axes[1, 2].lines + iteration_axis.lines
    axes[1, 2].legend(lines, [line.get_label() for line in lines], loc="best")
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
