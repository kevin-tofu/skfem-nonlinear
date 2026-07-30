"""Advection--diffusion on a moving ALE mesh.

The manufactured physical field is stationary.  Its variation observed by
moving mesh nodes is cancelled by the ALE relative velocity u_f - w_m.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementVector,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import ddot, div, dot, grad, sym_grad


DIFFUSIVITY = 2.0e-3
MAX_AMPLITUDE = 0.18
TIME_STEP = 5.0e-3
FINAL_TIME = 0.50


@BilinearForm
def pseudo_elastic_form(u, v, w):
    return 2.0 * ddot(sym_grad(u), sym_grad(v)) + div(u) * div(v)


@BilinearForm
def mass_form(u, v, w):
    return u * v


@BilinearForm
def diffusion_form(u, v, w):
    return DIFFUSIVITY * dot(grad(u), grad(v))


@BilinearForm
def advection_form(u, v, w):
    return dot(w.relative_velocity, grad(u)) * v


@LinearForm
def source_form(v, w):
    exact = np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1])
    return 2.0 * DIFFUSIVITY * np.pi**2 * exact * v


def exact_solution(points):
    return np.sin(np.pi * points[0]) * np.sin(np.pi * points[1])


def amplitude(time):
    return MAX_AMPLITUDE * np.sin(2.0 * np.pi * time)


def main() -> None:
    reference_mesh = MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 25),
        np.linspace(0.0, 1.0, 19),
    )
    vector_basis = Basis(reference_mesh, ElementVector(ElementTriP1()))
    boundary_vector = vector_basis.get_dofs().all()
    free_vector = vector_basis.complement_dofs(boundary_vector)
    top_vertical = vector_basis.get_dofs(
        lambda x: np.isclose(x[1], 1.0)
    ).all("u^2")
    prescribed = np.zeros(vector_basis.N)
    prescribed[top_vertical] = -np.sin(
        np.pi * vector_basis.doflocs[0, top_vertical]
    ) ** 2
    elastic_matrix = asm(pseudo_elastic_form, vector_basis).tocsr()
    unit_motion = prescribed.copy()
    unit_motion[free_vector] = spsolve(
        elastic_matrix[free_vector][:, free_vector],
        -elastic_matrix[free_vector][:, boundary_vector]
        @ prescribed[boundary_vector],
    )
    nodal_unit_motion = unit_motion[vector_basis.nodal_dofs]

    times = np.arange(0.0, FINAL_TIME + 0.5 * TIME_STEP, TIME_STEP)
    initial_points = reference_mesh.p + amplitude(0.0) * nodal_unit_motion
    initial_mesh = MeshTri(initial_points, reference_mesh.t.copy())
    initial_basis = Basis(initial_mesh, ElementTriP1())
    probe = np.argmin(
        np.sum(
            (reference_mesh.p - np.array([[0.5], [0.5]])) ** 2,
            axis=0,
        )
    )
    solutions = {
        "ALE relative velocity": exact_solution(initial_basis.doflocs),
        "mesh velocity omitted": exact_solution(initial_basis.doflocs),
    }
    error_history = {label: [0.0] for label in solutions}
    probe_history = {
        label: [solutions[label][probe]]
        for label in solutions
    }
    mesh_snapshot = None

    for step in range(1, times.size):
        old_amplitude = amplitude(times[step - 1])
        new_amplitude = amplitude(times[step])
        points = reference_mesh.p + new_amplitude * nodal_unit_motion
        mesh = MeshTri(points, reference_mesh.t.copy())
        basis = Basis(mesh, ElementTriP1())
        velocity_basis = Basis(mesh, ElementVector(ElementTriP1()))
        nodal_velocity = (
            (new_amplitude - old_amplitude) / TIME_STEP
        ) * nodal_unit_motion
        velocity_coefficients = np.zeros(velocity_basis.N)
        velocity_coefficients[velocity_basis.nodal_dofs] = nodal_velocity
        mesh_velocity = velocity_basis.interpolate(velocity_coefficients)

        mass = asm(mass_form, basis).tocsr()
        diffusion = asm(diffusion_form, basis).tocsr()
        source = asm(source_form, basis)
        boundary = basis.get_dofs().all()
        free = basis.complement_dofs(boundary)
        prescribed_scalar = exact_solution(basis.doflocs)
        exact = prescribed_scalar.copy()

        for label, relative_velocity in (
            ("ALE relative velocity", -mesh_velocity),
            ("mesh velocity omitted", 0.0 * mesh_velocity),
        ):
            advection = asm(
                advection_form,
                basis,
                relative_velocity=relative_velocity,
            ).tocsr()
            system = mass / TIME_STEP + advection + diffusion
            right_hand_side = mass @ (solutions[label] / TIME_STEP) + source
            updated = prescribed_scalar.copy()
            updated[free] = spsolve(
                system[free][:, free],
                right_hand_side[free]
                - system[free][:, boundary] @ prescribed_scalar[boundary],
            )
            solutions[label] = updated
            difference = updated - exact
            relative_error = np.sqrt(difference @ (mass @ difference)) / np.sqrt(
                exact @ (mass @ exact)
            )
            error_history[label].append(relative_error)
            probe_history[label].append(updated[probe])

        if abs(times[step] - 0.25) < 0.5 * TIME_STEP:
            mesh_snapshot = points.copy()

    assert mesh_snapshot is not None
    assert error_history["ALE relative velocity"][-1] < 0.02
    assert (
        error_history["ALE relative velocity"][-1]
        < 0.35 * error_history["mesh velocity omitted"][-1]
    )

    final_points = reference_mesh.p
    triangulation = mtri.Triangulation(
        final_points[0], final_points[1], reference_mesh.t.T
    )
    exact_final = exact_solution(final_points)
    moving_probe_points = (
        reference_mesh.p[:, probe, None]
        + amplitude(times)[None, :] * nodal_unit_motion[:, probe, None]
    )
    exact_probe_history = exact_solution(moving_probe_points)
    colors = {
        "ALE relative velocity": "#2c7fb8",
        "mesh velocity omitted": "#d95f0e",
    }
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.0))
    snapshot_triangulation = mtri.Triangulation(
        mesh_snapshot[0], mesh_snapshot[1], reference_mesh.t.T
    )
    axes[0, 0].triplot(snapshot_triangulation, color="0.25", linewidth=0.4)
    axes[0, 0].set_title("Moving mesh at maximum displacement")
    axes[0, 0].set_xlabel("$x$")
    axes[0, 0].set_ylabel("$y$")
    axes[0, 0].set_aspect("equal")

    plots = (
        ("ALE relative velocity", "Correct ALE solution"),
        ("mesh velocity omitted", "Without mesh-velocity correction"),
    )
    for axis, (label, title) in zip(axes[0, 1:], plots):
        field_plot = axis.tripcolor(
            triangulation,
            solutions[label],
            shading="gouraud",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        figure.colorbar(field_plot, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.set_aspect("equal")

    ale_error = solutions["ALE relative velocity"] - exact_final
    omitted_error = solutions["mesh velocity omitted"] - exact_final
    limit = max(np.max(np.abs(ale_error)), np.max(np.abs(omitted_error)))
    error_plot = axes[1, 0].tripcolor(
        triangulation,
        omitted_error,
        shading="gouraud",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    figure.colorbar(error_plot, ax=axes[1, 0], label="$c_h-c_{exact}$")
    axes[1, 0].set_title("Error when $w_m$ is omitted")
    axes[1, 0].set_xlabel("$x$")
    axes[1, 0].set_ylabel("$y$")
    axes[1, 0].set_aspect("equal")

    for label in solutions:
        axes[1, 1].semilogy(
            times,
            np.maximum(error_history[label], 1.0e-14),
            label=label,
            color=colors[label],
        )
    axes[1, 2].plot(
        times,
        exact_probe_history,
        "k:",
        linewidth=1.4,
        label="exact at moving node",
    )
    axes[1, 2].plot(
            times,
            probe_history[label],
            label=label,
            color=colors[label],
        )
    axes[1, 1].set(
        xlabel="time",
        ylabel="relative $L^2$ error",
        title="Manufactured-solution error",
    )
    axes[1, 2].plot(
        times,
        1.0 + amplitude(times),
        "k--",
        linewidth=1.0,
        label="top-interface height",
    )
    axes[1, 2].set(
        xlabel="time",
        ylabel="value",
        title="Scalar at a moving mesh node",
    )
    for axis in axes[1, 1:]:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle(
        r"ALE advection--diffusion: "
        r"$\partial_t c|_X+(u_f-w_m)\cdot\nabla c-D\Delta c=Q$",
        fontsize=14,
    )
    figure.tight_layout()
    output = Path(__file__).with_name("result.png")
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print("ALE advection--diffusion manufactured solution")
    for label in solutions:
        print(f"  {label:24s}: final relative L2 error = "
              f"{error_history[label][-1]:.6e}")
    print(f"  error ratio (omitted / ALE) = "
          f"{error_history['mesh velocity omitted'][-1] / error_history['ALE relative velocity'][-1]:.2f}")
    print(f"  wrote {output}")


if __name__ == "__main__":
    main()
