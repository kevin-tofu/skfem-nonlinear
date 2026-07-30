"""Strongly coupled partitioned ALE fluid--structure interaction."""

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
    FacetBasis,
    Functional,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import ddot, div, grad, sym_grad


VISCOSITY = 3.0e-2
STRUCTURE_MASS = 0.20
STRUCTURE_DAMPING = 0.08
STRUCTURE_STIFFNESS = 8.0
TIME_STEP = 0.02
FINAL_TIME = 0.80
COUPLING_RELAXATION = 0.55


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
def convection_form(u, v, w):
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


@Functional
def generalized_pressure_force(w):
    mode = np.sin(2.0 * np.pi * w.x[0])
    return w.pressure * w.n[1] * mode


def main() -> None:
    reference_mesh = MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 11),
        np.linspace(0.0, 1.0, 11),
    )
    top_facets = reference_mesh.facets_satisfying(
        lambda x: np.isclose(x[1], 1.0)
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
    nv = reference_velocity_basis.N
    reference_pressure_basis = Basis(
        reference_mesh, ElementTriP1(), intorder=5
    )
    np_ = reference_pressure_basis.N
    velocity_boundary = reference_velocity_basis.get_dofs().all()
    velocity_free = reference_velocity_basis.complement_dofs(
        velocity_boundary
    )
    pressure_free = np.arange(1, np_)
    free = np.concatenate((velocity_free, nv + pressure_free))
    fixed = np.setdiff1d(np.arange(nv + np_), free)
    top_vertical = reference_velocity_basis.get_dofs(
        lambda x: np.isclose(x[1], 1.0)
    ).all("u^2")
    mode_at_top_dofs = np.sin(
        2.0 * np.pi
        * reference_velocity_basis.doflocs[0, top_vertical]
    )
    zero_pressure = csr_matrix((np_, np_))

    def fluid_solve(
        interface_position,
        old_interface_position,
        old_state,
        initial_state,
    ):
        points = reference_mesh.p + interface_position * nodal_motion
        old_points = reference_mesh.p + old_interface_position * nodal_motion
        moving_mesh = MeshTri(points, reference_mesh.t.copy())
        old_mesh = MeshTri(old_points, reference_mesh.t.copy())
        velocity_basis = Basis(
            moving_mesh, ElementVector(ElementTriP2()), intorder=5
        )
        old_velocity_basis = Basis(
            old_mesh, ElementVector(ElementTriP2()), intorder=5
        )
        pressure_basis = Basis(moving_mesh, ElementTriP1(), intorder=5)
        mesh_velocity_vector = np.zeros(nv)
        for component, component_dofs in enumerate(
            velocity_basis.split_indices()
        ):
            mesh_velocity_vector[component_dofs] = (
                velocity_basis.doflocs[component, component_dofs]
                - old_velocity_basis.doflocs[component, component_dofs]
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
        right_hand_side = np.concatenate(
            (mass @ (old_state[:nv] / TIME_STEP), np.zeros(np_))
        )
        prescribed = np.zeros(nv + np_)
        prescribed[top_vertical] = (
            (interface_position - old_interface_position) / TIME_STEP
        ) * mode_at_top_dofs
        state = initial_state.copy()
        state[fixed] = prescribed[fixed]

        for nonlinear_iteration in range(1, 21):
            fluid_velocity = velocity_basis.interpolate(state[:nv])
            convection = asm(
                convection_form,
                velocity_basis,
                advector=fluid_velocity - mesh_velocity,
            ).tocsr()
            velocity_block = mass / TIME_STEP + viscous + convection
            system = bmat(
                [
                    [velocity_block, pressure_coupling],
                    [divergence_coupling, zero_pressure],
                ],
                format="csr",
            )
            candidate = prescribed.copy()
            candidate[free] = spsolve(
                system[free][:, free],
                right_hand_side[free]
                - system[free][:, fixed] @ prescribed[fixed],
            )
            change = np.linalg.norm(candidate - state) / max(
                np.linalg.norm(candidate), 1.0
            )
            state = 0.75 * candidate + 0.25 * state
            state[fixed] = prescribed[fixed]
            if change < 1.0e-8:
                break
        else:
            raise RuntimeError("fluid Picard iteration failed")

        top_pressure_basis = FacetBasis(
            moving_mesh,
            ElementTriP1(),
            facets=top_facets,
            intorder=5,
        )
        pressure_force = asm(
            generalized_pressure_force,
            top_pressure_basis,
            pressure=top_pressure_basis.interpolate(state[nv:]),
        )
        continuity = asm(
            continuity_residual,
            pressure_basis,
            velocity=velocity_basis.interpolate(state[:nv]),
        )
        return (
            state,
            float(pressure_force),
            points,
            mass,
            np.linalg.norm(continuity[pressure_free]),
            nonlinear_iteration,
        )

    times = np.arange(0.0, FINAL_TIME + 0.5 * TIME_STEP, TIME_STEP)
    interface_position = 0.040
    interface_velocity = 0.0
    fluid_state = np.zeros(nv + np_)
    position_history = [interface_position]
    velocity_history = [interface_velocity]
    pressure_force_history = [0.0]
    coupling_iterations = [0]
    coupling_residuals = [0.0]
    fluid_iterations = [0]
    continuity_history = [0.0]
    fluid_energy = [0.0]
    structure_energy = [
        0.5 * STRUCTURE_STIFFNESS * interface_position**2
    ]
    snapshots = {}

    effective_structure_stiffness = (
        STRUCTURE_MASS / TIME_STEP**2
        + STRUCTURE_DAMPING / TIME_STEP
        + STRUCTURE_STIFFNESS
    )
    for step in range(1, times.size):
        old_position = interface_position
        old_velocity = interface_velocity
        old_fluid_state = fluid_state.copy()
        trial_position = old_position + TIME_STEP * old_velocity
        trial_fluid_state = old_fluid_state.copy()
        residual_history = []

        for coupling_iteration in range(1, 41):
            (
                candidate_fluid_state,
                pressure_force,
                points,
                mass,
                continuity_norm,
                nonlinear_iteration,
            ) = fluid_solve(
                trial_position,
                old_position,
                old_fluid_state,
                trial_fluid_state,
            )
            structure_right_hand_side = (
                pressure_force
                + STRUCTURE_MASS * old_position / TIME_STEP**2
                + STRUCTURE_MASS * old_velocity / TIME_STEP
                + STRUCTURE_DAMPING * old_position / TIME_STEP
            )
            raw_position = (
                structure_right_hand_side / effective_structure_stiffness
            )
            coupling_residual = abs(raw_position - trial_position)
            residual_history.append(coupling_residual)
            trial_fluid_state = candidate_fluid_state
            if coupling_residual < 2.0e-7:
                trial_position = raw_position
                break
            trial_position += COUPLING_RELAXATION * (
                raw_position - trial_position
            )
        else:
            raise RuntimeError(f"FSI coupling failed at step {step}")

        # One final fluid solve makes the stored fluid state consistent with
        # the converged interface position.
        (
            fluid_state,
            pressure_force,
            points,
            mass,
            continuity_norm,
            nonlinear_iteration,
        ) = fluid_solve(
            trial_position,
            old_position,
            old_fluid_state,
            trial_fluid_state,
        )
        interface_position = trial_position
        interface_velocity = (
            interface_position - old_position
        ) / TIME_STEP
        position_history.append(interface_position)
        velocity_history.append(interface_velocity)
        pressure_force_history.append(pressure_force)
        coupling_iterations.append(coupling_iteration)
        coupling_residuals.append(residual_history[-1])
        fluid_iterations.append(nonlinear_iteration)
        continuity_history.append(continuity_norm)
        fluid_energy.append(
            0.5 * fluid_state[:nv] @ (mass @ fluid_state[:nv])
        )
        structure_energy.append(
            0.5 * STRUCTURE_MASS * interface_velocity**2
            + 0.5 * STRUCTURE_STIFFNESS * interface_position**2
        )

        if step in (1, times.size // 3, 2 * times.size // 3):
            snapshots[step] = (points.copy(), fluid_state.copy())

    assert max(continuity_history) < 1e-8
    assert max(coupling_residuals) < 2.0e-7
    assert max(coupling_iterations) < 40

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for axis, (step, (points, state)) in zip(axes[0], snapshots.items()):
        triangulation = mtri.Triangulation(
            points[0], points[1], reference_mesh.t.T
        )
        velocity_basis = Basis(
            MeshTri(points, reference_mesh.t.copy()),
            ElementVector(ElementTriP2()),
            intorder=5,
        )
        nodal_velocity = state[:nv][velocity_basis.nodal_dofs]
        speed = np.linalg.norm(nodal_velocity, axis=0)
        speed_plot = axis.tricontourf(
            triangulation, speed, levels=20, cmap="viridis"
        )
        figure.colorbar(speed_plot, ax=axis, label="speed")
        stride = 5
        axis.quiver(
            points[0, ::stride],
            points[1, ::stride],
            nodal_velocity[0, ::stride],
            nodal_velocity[1, ::stride],
            color="white",
            scale=3.0,
            width=0.004,
        )
        axis.set_title(f"FSI state at $t={times[step]:.2f}$")
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.set_aspect("equal")

    axes[1, 0].plot(
        times, position_history, label="modal displacement", color="#2c7fb8"
    )
    axes[1, 0].plot(
        times, velocity_history, label="modal velocity", color="#d95f0e"
    )
    axes[1, 0].set(
        xlabel="time", ylabel="interface state", title="Flexible-wall response"
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        times,
        pressure_force_history,
        label="fluid pressure force",
        color="#756bb1",
    )
    axes[1, 1].plot(
        times,
        structure_energy,
        label="structure energy",
        color="#d95f0e",
    )
    axes[1, 1].plot(
        times, fluid_energy, label="fluid kinetic energy", color="#2c7fb8"
    )
    axes[1, 1].set(
        xlabel="time", ylabel="force / energy", title="Load and energy exchange"
    )
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].step(
        times,
        coupling_iterations,
        where="post",
        label="FSI coupling",
        color="#2c7fb8",
    )
    axes[1, 2].step(
        times,
        fluid_iterations,
        where="post",
        label="fluid Picard",
        color="#d95f0e",
    )
    axes[1, 2].set(
        xlabel="time", ylabel="iterations", title="Nested nonlinear iterations"
    )
    axes[1, 2].legend(fontsize=8)
    for axis in axes[1]:
        axis.grid(alpha=0.25)

    figure.suptitle(
        "Partitioned ALE FSI: pressure-loaded flexible wall "
        f"(relaxation={COUPLING_RELAXATION:.2f})",
        fontsize=14,
    )
    figure.tight_layout()
    output = Path(__file__).with_name("result.png")
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print("Strongly coupled partitioned ALE FSI")
    print(f"  time steps = {times.size - 1}, dt = {TIME_STEP:g}")
    print(f"  displacement range = "
          f"[{min(position_history):.6f}, {max(position_history):.6f}]")
    print(f"  maximum FSI iterations = {max(coupling_iterations)}")
    print(f"  maximum fluid Picard iterations = {max(fluid_iterations)}")
    print(f"  maximum coupling residual = {max(coupling_residuals):.3e}")
    print(f"  maximum continuity residual = {max(continuity_history):.3e}")
    print(f"  wrote {output}")


if __name__ == "__main__":
    main()
