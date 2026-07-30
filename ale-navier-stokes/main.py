"""Transient incompressible Navier--Stokes on a moving ALE mesh."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import ddot, div, grad, sym_grad


VISCOSITY = 2.0e-2
MOTION_SCALE = 0.06
TIME_STEP = 0.02
FINAL_TIME = 1.0


@BilinearForm
def mesh_elasticity(u, v, w):
    return 2.0 * ddot(sym_grad(u), sym_grad(v)) + div(u) * div(v)


@BilinearForm
def velocity_mass(u, v, w):
    return np.einsum("i...,i...->...", u, v)


@BilinearForm
def viscous_form(u, v, w):
    return VISCOSITY * ddot(grad(u), grad(v))


@BilinearForm
def picard_convection(u, v, w):
    return np.einsum("j...,ij...,i...->...", w.advector, grad(u), v)


@BilinearForm
def pressure_to_momentum(p, v, w):
    return -p * div(v)


@BilinearForm
def velocity_to_continuity(u, q, w):
    return q * div(u)


@LinearForm
def continuity_residual(q, w):
    return q * div(w.velocity)


def amplitude(time):
    return MOTION_SCALE * (1.0 - np.cos(2.0 * np.pi * time))


def main() -> None:
    reference_mesh = MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 13),
        np.linspace(0.0, 1.0, 13),
    )
    mesh_basis = Basis(reference_mesh, ElementVector(ElementTriP1()))
    mesh_boundary = mesh_basis.get_dofs().all()
    mesh_free = mesh_basis.complement_dofs(mesh_boundary)
    mesh_top_vertical = mesh_basis.get_dofs(
        lambda x: np.isclose(x[1], 1.0)
    ).all("u^2")
    prescribed_motion = np.zeros(mesh_basis.N)
    prescribed_motion[mesh_top_vertical] = np.sin(
        2.0 * np.pi * mesh_basis.doflocs[0, mesh_top_vertical]
    )
    mesh_matrix = asm(mesh_elasticity, mesh_basis).tocsr()
    unit_motion = prescribed_motion.copy()
    unit_motion[mesh_free] = spsolve(
        mesh_matrix[mesh_free][:, mesh_free],
        -mesh_matrix[mesh_free][:, mesh_boundary]
        @ prescribed_motion[mesh_boundary],
    )
    nodal_motion = unit_motion[mesh_basis.nodal_dofs]

    reference_velocity_basis = Basis(
        reference_mesh, ElementVector(ElementTriP2()), intorder=5
    )
    reference_pressure_basis = Basis(
        reference_mesh, ElementTriP1(), intorder=5
    )
    nv = reference_velocity_basis.N
    np_ = reference_pressure_basis.N
    velocity_boundary = reference_velocity_basis.get_dofs().all()
    velocity_free = reference_velocity_basis.complement_dofs(velocity_boundary)
    pressure_free = np.arange(1, np_)
    free = np.concatenate((velocity_free, nv + pressure_free))
    fixed = np.setdiff1d(np.arange(nv + np_), free)
    top_vertical = reference_velocity_basis.get_dofs(
        lambda x: np.isclose(x[1], 1.0)
    ).all("u^2")
    zero_pressure = csr_matrix((np_, np_))

    states = {
        "ALE relative velocity": np.zeros(nv + np_),
        "mesh velocity omitted": np.zeros(nv + np_),
    }
    colors = {
        "ALE relative velocity": "#2c7fb8",
        "mesh velocity omitted": "#d95f0e",
    }
    times = np.arange(0.0, FINAL_TIME + 0.5 * TIME_STEP, TIME_STEP)
    kinetic_energy = {label: [0.0] for label in states}
    pressure_peak = {label: [0.0] for label in states}
    iteration_history = {label: [0] for label in states}
    divergence_history = {label: [0.0] for label in states}
    snapshots = {}

    old_points = reference_mesh.p.copy()
    for step in range(1, times.size):
        new_points = reference_mesh.p + amplitude(times[step]) * nodal_motion
        moving_mesh = MeshTri(new_points, reference_mesh.t.copy())
        velocity_basis = Basis(
            moving_mesh, ElementVector(ElementTriP2()), intorder=5
        )
        pressure_basis = Basis(moving_mesh, ElementTriP1(), intorder=5)
        old_geometry_basis = Basis(
            MeshTri(old_points, reference_mesh.t.copy()),
            ElementVector(ElementTriP2()),
            intorder=5,
        )
        mesh_velocity_vector = np.zeros(nv)
        for component, component_dofs in enumerate(
            velocity_basis.split_indices()
        ):
            mesh_velocity_vector[component_dofs] = (
                velocity_basis.doflocs[component, component_dofs]
                - old_geometry_basis.doflocs[component, component_dofs]
            ) / TIME_STEP
        mesh_velocity = velocity_basis.interpolate(mesh_velocity_vector)

        mass = asm(velocity_mass, velocity_basis).tocsr()
        viscous = asm(viscous_form, velocity_basis).tocsr()
        pressure_coupling = asm(
            pressure_to_momentum, pressure_basis, velocity_basis
        )
        divergence_coupling = asm(
            velocity_to_continuity, velocity_basis, pressure_basis
        )
        prescribed_state = np.zeros(nv + np_)
        prescribed_state[top_vertical] = (
            amplitude(times[step]) - amplitude(times[step - 1])
        ) / TIME_STEP * np.sin(
            2.0 * np.pi
            * reference_velocity_basis.doflocs[0, top_vertical]
        )

        for label, old_state in list(states.items()):
            state = old_state.copy()
            state[fixed] = prescribed_state[fixed]
            right_momentum = mass @ (old_state[:nv] / TIME_STEP)
            for iteration in range(1, 31):
                fluid_velocity = velocity_basis.interpolate(state[:nv])
                advector = (
                    fluid_velocity - mesh_velocity
                    if label == "ALE relative velocity"
                    else fluid_velocity
                )
                convection = asm(
                    picard_convection,
                    velocity_basis,
                    advector=advector,
                ).tocsr()
                velocity_block = mass / TIME_STEP + viscous + convection
                system = bmat(
                    [
                        [velocity_block, pressure_coupling],
                        [divergence_coupling, zero_pressure],
                    ],
                    format="csr",
                )
                right_hand_side = np.concatenate(
                    (right_momentum, np.zeros(np_))
                )
                candidate = prescribed_state.copy()
                candidate[free] = spsolve(
                    system[free][:, free],
                    right_hand_side[free]
                    - system[free][:, fixed] @ prescribed_state[fixed],
                )
                relative_change = np.linalg.norm(candidate - state) / max(
                    np.linalg.norm(candidate), 1.0
                )
                state = 0.7 * candidate + 0.3 * state
                state[fixed] = prescribed_state[fixed]
                if relative_change < 2.0e-9:
                    break
            else:
                raise RuntimeError(f"{label} Picard failed at step {step}")

            states[label] = state
            kinetic_energy[label].append(
                0.5 * state[:nv] @ (mass @ state[:nv])
            )
            pressure_peak[label].append(np.max(np.abs(state[nv:])))
            iteration_history[label].append(iteration)
            continuity = asm(
                continuity_residual,
                pressure_basis,
                velocity=velocity_basis.interpolate(state[:nv]),
            )
            divergence_history[label].append(
                np.linalg.norm(continuity[pressure_free])
            )

        if abs(times[step] - 0.50) < 0.5 * TIME_STEP:
            snapshots = {
                "points": new_points.copy(),
                "states": {label: state.copy() for label, state in states.items()},
            }
        old_points = new_points

    assert snapshots
    assert max(max(values) for values in divergence_history.values()) < 1e-9
    final_difference = np.linalg.norm(
        states["ALE relative velocity"][:nv]
        - states["mesh velocity omitted"][:nv]
    ) / np.linalg.norm(states["ALE relative velocity"][:nv])
    assert final_difference > 1e-3

    snapshot_mesh = MeshTri(snapshots["points"], reference_mesh.t.copy())
    snapshot_velocity_basis = Basis(
        snapshot_mesh, ElementVector(ElementTriP2()), intorder=5
    )
    triangulation = mtri.Triangulation(
        snapshots["points"][0],
        snapshots["points"][1],
        reference_mesh.t.T,
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for axis, label in zip(axes[0, :2], states):
        snapshot_state = snapshots["states"][label]
        nodal_velocity = snapshot_state[:nv][
            snapshot_velocity_basis.nodal_dofs
        ]
        speed = np.linalg.norm(nodal_velocity, axis=0)
        speed_plot = axis.tricontourf(
            triangulation, speed, levels=24, cmap="viridis"
        )
        figure.colorbar(speed_plot, ax=axis, label="speed")
        stride = 6
        axis.quiver(
            snapshots["points"][0, ::stride],
            snapshots["points"][1, ::stride],
            nodal_velocity[0, ::stride],
            nodal_velocity[1, ::stride],
            color="white",
            scale=4.0,
            width=0.004,
        )
        axis.set_title(label)
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.set_aspect("equal")

    pressure = snapshots["states"]["ALE relative velocity"][nv:]
    pressure_plot = axes[0, 2].tricontourf(
        triangulation, pressure, levels=24, cmap="coolwarm"
    )
    figure.colorbar(pressure_plot, ax=axes[0, 2], label="pressure")
    axes[0, 2].set_title("ALE pressure at maximum deformation")
    axes[0, 2].set_xlabel("$x$")
    axes[0, 2].set_ylabel("$y$")
    axes[0, 2].set_aspect("equal")

    for label in states:
        axes[1, 0].plot(
            times, kinetic_energy[label], label=label, color=colors[label]
        )
        axes[1, 1].plot(
            times, pressure_peak[label], label=label, color=colors[label]
        )
        axes[1, 2].plot(
            times,
            iteration_history[label],
            label=label,
            color=colors[label],
        )
    axes[1, 0].set(
        xlabel="time", ylabel="kinetic energy", title="Fluid kinetic energy"
    )
    axes[1, 1].set(
        xlabel="time", ylabel="$\\max |p|$", title="Pressure response"
    )
    axes[1, 2].set(
        xlabel="time", ylabel="Picard iterations", title="Nonlinear iterations"
    )
    for axis in axes[1]:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        r"Moving-wall ALE Navier--Stokes: "
        r"$(u-w_m)\cdot\nabla u$, Taylor--Hood $P_2/P_1$",
        fontsize=14,
    )
    figure.tight_layout()
    output = Path(__file__).with_name("result.png")
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print("ALE incompressible Navier--Stokes")
    print(f"  time steps = {times.size - 1}, dt = {TIME_STEP:g}")
    print(f"  maximum mesh displacement = {2.0 * MOTION_SCALE:.3f}")
    print(f"  final velocity difference = {final_difference:.6e}")
    for label in states:
        print(
            f"  {label:24s}: max Picard = "
            f"{max(iteration_history[label])}, max continuity residual = "
            f"{max(divergence_history[label]):.3e}"
        )
    print(f"  wrote {output}")


if __name__ == "__main__":
    main()
