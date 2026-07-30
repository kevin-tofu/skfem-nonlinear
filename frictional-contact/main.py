"""Nodal Coulomb friction against a curved rigid obstacle."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, ElementTriP1, ElementVector, MeshTri, asm
from skfem.models.elasticity import linear_elasticity


YOUNG, POISSON = 1000.0, 0.30
LAME_LAMBDA = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
LAME_MU = YOUNG / (2.0 * (1.0 + POISSON))
NORMAL_PENALTY = 2.0e5
TANGENTIAL_PENALTY = 5.0e4
FRICTION = 0.35


def obstacle_gap(x):
    return 0.012 + 0.08 * (x - 0.5) ** 2


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 33), np.linspace(0.0, 1.0, 17))
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    stiffness = asm(linear_elasticity(LAME_LAMBDA, LAME_MU), basis).tocsr()
    top = basis.get_dofs(lambda x: np.isclose(x[1], 1.0))
    bottom = basis.get_dofs(lambda x: np.isclose(x[1], 0.0))
    top_x, top_y = top.nodal["u^1"], top.nodal["u^2"]
    bottom_x, bottom_y = bottom.nodal["u^1"], bottom.nodal["u^2"]
    fixed = np.unique(np.concatenate((top_x, top_y)))
    free = basis.complement_dofs(fixed)
    gaps = obstacle_gap(basis.doflocs[0, bottom_y])

    vertical_path = [(0.0, value) for value in -np.linspace(0.01, 0.055, 6)]
    horizontal_values = np.concatenate(
        (
            np.linspace(0.0, 0.06, 13)[1:],
            np.linspace(0.055, -0.06, 24),
            np.linspace(-0.055, 0.0, 12),
        )
    )
    load_path = vertical_path + [(value, -0.055) for value in horizontal_values]

    displacement = np.zeros(basis.N)
    committed_slip = np.zeros(bottom_x.size)
    horizontal_history = []
    shear_reaction_history = []
    friction_force_history = []
    normal_reaction_history = []
    stick_history = []
    slip_history = []
    iteration_history = []

    def contact_response(state, slip):
        gap = state[bottom_y] + gaps
        active = gap < 0.0
        normal_force = np.where(active, -NORMAL_PENALTY * gap, 0.0)
        trial_tangent = TANGENTIAL_PENALTY * (state[bottom_x] - slip)
        limit = FRICTION * normal_force
        sticking = active & (np.abs(trial_tangent) <= limit)
        sliding = active & ~sticking
        tangent_force = np.zeros_like(trial_tangent)
        tangent_force[sticking] = trial_tangent[sticking]
        tangent_force[sliding] = (
            limit[sliding] * np.sign(trial_tangent[sliding])
        )
        return gap, normal_force, tangent_force, active, sticking, sliding

    for step, (imposed_x, imposed_y) in enumerate(load_path, 1):
        displacement[top_x] = imposed_x
        displacement[top_y] = imposed_y
        for iteration in range(1, 41):
            (
                gap,
                normal_force,
                tangent_force,
                active,
                sticking,
                sliding,
            ) = contact_response(displacement, committed_slip)
            residual = stiffness @ displacement
            residual[bottom_y] -= normal_force
            residual[bottom_x] += tangent_force
            norm_residual = np.linalg.norm(residual[free])
            if norm_residual < 1.0e-8:
                break

            rows = []
            columns = []
            values = []
            for local in np.flatnonzero(active):
                rows.append(bottom_y[local])
                columns.append(bottom_y[local])
                values.append(NORMAL_PENALTY)
            for local in np.flatnonzero(sticking):
                rows.append(bottom_x[local])
                columns.append(bottom_x[local])
                values.append(TANGENTIAL_PENALTY)
            for local in np.flatnonzero(sliding):
                rows.append(bottom_x[local])
                columns.append(bottom_y[local])
                values.append(
                    -FRICTION
                    * NORMAL_PENALTY
                    * np.sign(tangent_force[local])
                )
            contact_tangent = coo_matrix(
                (values, (rows, columns)), shape=stiffness.shape
            ).tocsr()
            tangent = stiffness + contact_tangent
            increment = np.zeros_like(displacement)
            increment[free] = spsolve(
                tangent[free][:, free], -residual[free]
            )

            alpha = 1.0
            while alpha > 1.0e-8:
                trial = displacement + alpha * increment
                trial_values = contact_response(trial, committed_slip)
                trial_residual = stiffness @ trial
                trial_residual[bottom_y] -= trial_values[1]
                trial_residual[bottom_x] += trial_values[2]
                if np.linalg.norm(trial_residual[free]) < norm_residual:
                    displacement = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(f"line search failed at step {step}")
        else:
            raise RuntimeError(f"contact Newton failed at step {step}")

        (
            gap,
            normal_force,
            tangent_force,
            active,
            sticking,
            sliding,
        ) = contact_response(displacement, committed_slip)
        committed_slip[sliding] = (
            displacement[bottom_x[sliding]]
            - tangent_force[sliding] / TANGENTIAL_PENALTY
        )
        structural_residual = stiffness @ displacement
        horizontal_history.append(imposed_x)
        shear_reaction_history.append(structural_residual[top_x].sum())
        friction_force_history.append(tangent_force.sum())
        normal_reaction_history.append(-structural_residual[top_y].sum())
        stick_history.append(np.count_nonzero(sticking))
        slip_history.append(np.count_nonzero(sliding))
        iteration_history.append(iteration)
        print(
            f"step={step:2d}, ux={imposed_x:+.4f}, uy={imposed_y:+.4f}, "
            f"normal={normal_force.sum():.3f}, friction={tangent_force.sum():+.3f}, "
            f"stick={np.count_nonzero(sticking):2d}, "
            f"slip={np.count_nonzero(sliding):2d}, Newton={iteration:2d}"
        )

    assert np.min(displacement[bottom_y] + gaps) > -5.0e-4
    assert max(slip_history) > 0
    assert np.max(np.abs(committed_slip)) > 0.0

    nodal_displacement = displacement[basis.nodal_dofs]
    deformed = mesh.p + nodal_displacement
    triangulation = mtri.Triangulation(deformed[0], deformed[1], mesh.t.T)
    bottom_coordinate = basis.doflocs[0, bottom_x]
    order = np.argsort(bottom_coordinate)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].triplot(triangulation, color="tab:blue", linewidth=0.5)
    axes[0, 0].plot(
        bottom_coordinate[order],
        -gaps[order],
        "k-",
        linewidth=2,
        label="rigid obstacle",
    )
    axes[0, 0].set(title="Final deformed contact state", xlabel="x", ylabel="y", aspect="equal")
    axes[0, 0].legend()

    axes[0, 1].plot(bottom_coordinate, normal_force, "o-", label="normal")
    axes[0, 1].plot(bottom_coordinate, tangent_force, "s-", label="tangential")
    axes[0, 1].set(title="Final contact-force distribution", xlabel="x", ylabel="nodal force")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].plot(bottom_coordinate, committed_slip, "o-")
    axes[0, 2].set(title="Accumulated frictional slip", xlabel="x", ylabel="slip")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(
        horizontal_history,
        shear_reaction_history,
        "o-",
        markersize=3,
        label="top reaction",
    )
    axes[1, 0].plot(
        horizontal_history,
        friction_force_history,
        "s-",
        markersize=3,
        label="contact friction",
    )
    axes[1, 0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1, 0].set(title="Friction hysteresis", xlabel="imposed horizontal displacement", ylabel="top shear reaction")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].stackplot(
        range(1, len(load_path) + 1),
        stick_history,
        slip_history,
        labels=("stick", "slip"),
        alpha=0.8,
    )
    axes[1, 1].set(title="Contact-state evolution", xlabel="load step", ylabel="contact nodes")
    axes[1, 1].legend(loc="upper left")

    axes[1, 2].step(range(1, len(load_path) + 1), iteration_history, where="mid")
    axes[1, 2].set(title="Nonsmooth Newton work", xlabel="load step", ylabel="iterations")
    axes[1, 2].grid(True, alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
