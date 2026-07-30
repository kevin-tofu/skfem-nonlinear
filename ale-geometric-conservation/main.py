"""Elementwise geometric conservation law for an ALE mesh."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, MeshTri, asm
from skfem.helpers import ddot, div, sym_grad


MAX_AMPLITUDE = 0.20
FINAL_TIME = 0.50
TIME_STEP = 0.005


@BilinearForm
def pseudo_elastic_form(u, v, w):
    return 2.0 * ddot(sym_grad(u), sym_grad(v)) + div(u) * div(v)


def amplitude(time):
    return MAX_AMPLITUDE * np.sin(2.0 * np.pi * time)


def element_matrices(points, connectivity):
    vertices = points[:, connectivity]
    return np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
        ),
        axis=2,
    ).transpose(1, 0, 2)


def determinants(matrices):
    return (
        matrices[:, 0, 0] * matrices[:, 1, 1]
        - matrices[:, 0, 1] * matrices[:, 1, 0]
    )


def main() -> None:
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 27),
        np.linspace(0.0, 1.0, 21),
    )
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    boundary = basis.get_dofs().all()
    free = basis.complement_dofs(boundary)
    top_vertical = basis.get_dofs(
        lambda x: np.isclose(x[1], 1.0)
    ).all("u^2")
    prescribed = np.zeros(basis.N)
    prescribed[top_vertical] = -np.sin(
        np.pi * basis.doflocs[0, top_vertical]
    ) ** 2
    stiffness = asm(pseudo_elastic_form, basis).tocsr()
    unit_motion = prescribed.copy()
    unit_motion[free] = spsolve(
        stiffness[free][:, free],
        -stiffness[free][:, boundary] @ prescribed[boundary],
    )
    nodal_motion = unit_motion[basis.nodal_dofs]

    reference_matrices = element_matrices(mesh.p, mesh.t)
    reference_determinants = determinants(reference_matrices)
    reference_areas = 0.5 * np.abs(reference_determinants)
    initial_density = np.ones(mesh.t.shape[1])
    initial_element_mass = initial_density * reference_areas
    initial_total_mass = np.sum(initial_element_mass)

    times = np.arange(0.0, FINAL_TIME + 0.5 * TIME_STEP, TIME_STEP)
    domain_area = []
    conservative_mass = []
    naive_mass = []
    minimum_jacobian = []
    maximum_gcl_residual = []
    density_at_maximum = None
    points_at_maximum = None
    jacobian_at_maximum = None

    for step, time in enumerate(times):
        points = mesh.p + amplitude(time) * nodal_motion
        current_matrices = element_matrices(points, mesh.t)
        current_determinants = determinants(current_matrices)
        current_areas = 0.5 * np.abs(current_determinants)
        conservative_density = initial_element_mass / current_areas
        domain_area.append(np.sum(current_areas))
        conservative_mass.append(np.sum(conservative_density * current_areas))
        naive_mass.append(np.sum(initial_density * current_areas))
        minimum_jacobian.append(
            np.min(current_determinants / reference_determinants)
        )

        if abs(time - 0.25) < 0.5 * TIME_STEP:
            density_at_maximum = conservative_density.copy()
            points_at_maximum = points.copy()
            jacobian_at_maximum = (
                current_determinants / reference_determinants
            )

        if step == 0:
            maximum_gcl_residual.append(0.0)
            continue

        old_points = mesh.p + amplitude(times[step - 1]) * nodal_motion
        old_matrices = element_matrices(old_points, mesh.t)
        midpoint_matrices = 0.5 * (old_matrices + current_matrices)
        matrix_velocity = (current_matrices - old_matrices) / TIME_STEP
        determinant_rate = (
            current_determinants - determinants(old_matrices)
        ) / TIME_STEP
        metric_rate = np.empty_like(determinant_rate)
        for element in range(mesh.t.shape[1]):
            metric_rate[element] = determinants(
                midpoint_matrices[element : element + 1]
            )[0] * np.trace(
                np.linalg.solve(
                    midpoint_matrices[element],
                    matrix_velocity[element],
                )
            )
        scale = np.maximum(np.abs(determinant_rate), 1.0)
        maximum_gcl_residual.append(
            np.max(np.abs(determinant_rate - metric_rate) / scale)
        )

    assert density_at_maximum is not None
    assert points_at_maximum is not None
    assert jacobian_at_maximum is not None
    conservative_mass = np.asarray(conservative_mass)
    naive_mass = np.asarray(naive_mass)
    maximum_gcl_residual = np.asarray(maximum_gcl_residual)
    assert np.max(np.abs(conservative_mass / initial_total_mass - 1.0)) < 1e-13
    assert np.max(maximum_gcl_residual) < 1e-11
    assert np.max(np.abs(naive_mass / initial_total_mass - 1.0)) > 0.05

    # A common inconsistent choice evaluates J div(w) at the end of a time
    # interval.  Midpoint metrics satisfy the discrete GCL for the piecewise
    # linear-in-time mesh trajectory, while endpoint metrics converge only
    # with the time step.
    time_steps = np.array([0.04, 0.02, 0.01, 0.005])
    midpoint_errors = []
    endpoint_errors = []
    for dt in time_steps:
        sample_times = np.arange(0.0, FINAL_TIME + 0.5 * dt, dt)
        midpoint_maximum = 0.0
        endpoint_maximum = 0.0
        for old_time, new_time in zip(sample_times[:-1], sample_times[1:]):
            old_points = mesh.p + amplitude(old_time) * nodal_motion
            new_points = mesh.p + amplitude(new_time) * nodal_motion
            old_matrices = element_matrices(old_points, mesh.t)
            new_matrices = element_matrices(new_points, mesh.t)
            midpoint_matrices = 0.5 * (old_matrices + new_matrices)
            matrix_velocity = (new_matrices - old_matrices) / dt
            rate = (determinants(new_matrices) - determinants(old_matrices)) / dt
            midpoint_rate = np.empty_like(rate)
            endpoint_rate = np.empty_like(rate)
            for element in range(mesh.t.shape[1]):
                midpoint_rate[element] = determinants(
                    midpoint_matrices[element : element + 1]
                )[0] * np.trace(
                    np.linalg.solve(
                        midpoint_matrices[element], matrix_velocity[element]
                    )
                )
                endpoint_rate[element] = determinants(
                    new_matrices[element : element + 1]
                )[0] * np.trace(
                    np.linalg.solve(
                        new_matrices[element], matrix_velocity[element]
                    )
                )
            midpoint_maximum = max(
                midpoint_maximum, np.max(np.abs(rate - midpoint_rate))
            )
            endpoint_maximum = max(
                endpoint_maximum, np.max(np.abs(rate - endpoint_rate))
            )
        midpoint_errors.append(midpoint_maximum)
        endpoint_errors.append(endpoint_maximum)

    triangulation = mtri.Triangulation(
        points_at_maximum[0], points_at_maximum[1], mesh.t.T
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.0))
    axes[0, 0].triplot(triangulation, color="0.25", linewidth=0.4)
    axes[0, 0].set_title("ALE mesh at maximum displacement")
    axes[0, 0].set_xlabel("$x$")
    axes[0, 0].set_ylabel("$y$")
    axes[0, 0].set_aspect("equal")

    jacobian_plot = axes[0, 1].tripcolor(
        triangulation, jacobian_at_maximum, shading="flat", cmap="viridis"
    )
    figure.colorbar(jacobian_plot, ax=axes[0, 1], label="$J_m/J_{m,0}$")
    axes[0, 1].set_title("Element Jacobian ratio")
    axes[0, 1].set_xlabel("$x$")
    axes[0, 1].set_ylabel("$y$")
    axes[0, 1].set_aspect("equal")

    density_plot = axes[0, 2].tripcolor(
        triangulation, density_at_maximum, shading="flat", cmap="magma"
    )
    figure.colorbar(density_plot, ax=axes[0, 2], label="cell density")
    axes[0, 2].set_title("GCL-consistent material density")
    axes[0, 2].set_xlabel("$x$")
    axes[0, 2].set_ylabel("$y$")
    axes[0, 2].set_aspect("equal")

    axes[1, 0].plot(
        times,
        conservative_mass / initial_total_mass,
        label="conservative metric update",
        color="#2c7fb8",
    )
    axes[1, 0].plot(
        times,
        naive_mass / initial_total_mass,
        label="density coefficients frozen",
        color="#d95f0e",
    )
    axes[1, 0].set(
        xlabel="time",
        ylabel="$M(t)/M(0)$",
        title="Total transported quantity",
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].semilogy(
        times[1:],
        np.maximum(maximum_gcl_residual[1:], 1e-17),
        color="#2c7fb8",
    )
    axes[1, 1].set(
        xlabel="time",
        ylabel="maximum element residual",
        title=r"Discrete GCL: $\dot J-J\,\nabla\cdot w_m$",
    )

    axes[1, 2].loglog(
        time_steps,
        np.maximum(midpoint_errors, 1e-17),
        "o-",
        label="midpoint metric",
        color="#2c7fb8",
    )
    axes[1, 2].loglog(
        time_steps,
        endpoint_errors,
        "o-",
        label="endpoint metric",
        color="#d95f0e",
    )
    axes[1, 2].set(
        xlabel="$\\Delta t$",
        ylabel="maximum absolute defect",
        title="Metric time-level consistency",
    )
    axes[1, 2].legend(fontsize=8)
    for axis in axes[1]:
        axis.grid(alpha=0.25)

    figure.suptitle(
        r"ALE geometric conservation: "
        r"$\dot J_m=J_m\,\nabla_x\cdot w_m$",
        fontsize=14,
    )
    figure.tight_layout()
    output = Path(__file__).with_name("result.png")
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print("ALE geometric conservation law")
    print(f"  maximum relative GCL residual = "
          f"{np.max(maximum_gcl_residual):.3e}")
    print(f"  conservative mass drift = "
          f"{np.max(np.abs(conservative_mass / initial_total_mass - 1.0)):.3e}")
    print(f"  frozen-density mass drift = "
          f"{np.max(np.abs(naive_mass / initial_total_mass - 1.0)):.3e}")
    print(f"  minimum mesh Jacobian ratio = {np.min(minimum_jacobian):.6f}")
    print(f"  wrote {output}")


if __name__ == "__main__":
    main()
