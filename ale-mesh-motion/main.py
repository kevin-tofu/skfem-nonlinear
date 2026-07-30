"""ALE mesh motion by harmonic and pseudo-elastic extensions."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, MeshTri, asm
from skfem.helpers import ddot, div, grad, sym_grad


MAX_AMPLITUDE = 0.28


@BilinearForm
def harmonic_form(u, v, w):
    """Independent Laplace smoothing of both displacement components."""
    return ddot(grad(u), grad(v))


@BilinearForm
def pseudo_elastic_form(u, v, w):
    """Fictitious linear elasticity used only as a mesh smoother."""
    lame_lambda, shear_modulus = 1.0, 1.0
    return (
        2.0 * shear_modulus * ddot(sym_grad(u), sym_grad(v))
        + lame_lambda * div(u) * div(v)
    )


def triangle_quality(points, connectivity, reference_area):
    """Return signed area ratio and minimum interior angle for every triangle."""
    vertices = points[:, connectivity]
    first = vertices[:, 1] - vertices[:, 0]
    second = vertices[:, 2] - vertices[:, 0]
    signed_area = first[0] * second[1] - first[1] * second[0]
    jacobian_ratio = signed_area / reference_area

    edge_lengths = np.stack(
        (
            np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=0),
            np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=0),
            np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=0),
        )
    )
    angles = []
    for opposite in range(3):
        adjacent_1 = edge_lengths[(opposite + 1) % 3]
        adjacent_2 = edge_lengths[(opposite + 2) % 3]
        cosine = (
            adjacent_1**2
            + adjacent_2**2
            - edge_lengths[opposite] ** 2
        ) / (2.0 * adjacent_1 * adjacent_2)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return jacobian_ratio, np.min(np.stack(angles), axis=0)


def draw_mesh(axis, points, connectivity, title):
    triangulation = mtri.Triangulation(points[0], points[1], connectivity.T)
    axis.triplot(triangulation, color="0.25", linewidth=0.35)
    axis.plot(points[0], points[1], ".", color="#2c7fb8", markersize=0.7)
    axis.set_aspect("equal")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(title)
    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")


def main() -> None:
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 33),
        np.linspace(0.0, 1.0, 25),
    )
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    boundary_dofs = basis.get_dofs().all()
    top_vertical_dofs = basis.get_dofs(
        lambda x: np.isclose(x[1], 1.0)
    ).all("u^2")
    prescribed_unit = np.zeros(basis.N)
    top_x = basis.doflocs[0, top_vertical_dofs]
    prescribed_unit[top_vertical_dofs] = -np.sin(np.pi * top_x) ** 2
    free_dofs = basis.complement_dofs(boundary_dofs)

    def extension(form):
        matrix = asm(form, basis).tocsr()
        solution = prescribed_unit.copy()
        solution[free_dofs] = spsolve(
            matrix[free_dofs][:, free_dofs],
            -matrix[free_dofs][:, boundary_dofs]
            @ prescribed_unit[boundary_dofs],
        )
        return solution

    unit_solutions = {
        "Harmonic": extension(harmonic_form),
        "Pseudo-elastic": extension(pseudo_elastic_form),
    }
    reference_area, _ = triangle_quality(
        mesh.p, mesh.t, np.ones(mesh.t.shape[1])
    )
    amplitudes = np.linspace(0.0, MAX_AMPLITUDE, 29)
    quality_history = {}
    deformed_points = {}

    for label, unit_solution in unit_solutions.items():
        nodal_unit = unit_solution[basis.nodal_dofs]
        minimum_jacobian = []
        minimum_angle = []
        for amplitude in amplitudes:
            points = mesh.p + amplitude * nodal_unit
            jacobian, angles = triangle_quality(points, mesh.t, reference_area)
            minimum_jacobian.append(np.min(jacobian))
            minimum_angle.append(np.min(angles))
        quality_history[label] = (
            np.asarray(minimum_jacobian),
            np.asarray(minimum_angle),
        )
        deformed_points[label] = mesh.p + MAX_AMPLITUDE * nodal_unit

    # At t = 0, A(t) = Amax sin(2 pi t) has zero displacement but maximum
    # mesh velocity.  This is the velocity appearing in (u_f - w_m) . grad u_f.
    mesh_velocity = (
        2.0
        * np.pi
        * MAX_AMPLITUDE
        * unit_solutions["Pseudo-elastic"][basis.nodal_dofs]
    )
    harmonic_jacobian, _ = triangle_quality(
        deformed_points["Harmonic"], mesh.t, reference_area
    )
    elastic_jacobian, _ = triangle_quality(
        deformed_points["Pseudo-elastic"], mesh.t, reference_area
    )

    # This deliberately reaches beyond the harmonic smoother's validity:
    # its first inverted element makes the quality monitor meaningful.
    assert np.min(harmonic_jacobian) < 0.0
    assert np.min(elastic_jacobian) > 0.0
    top_velocity_error = np.max(
        np.abs(
            mesh_velocity[1, mesh.p[1] == 1.0]
            + 2.0
            * np.pi
            * MAX_AMPLITUDE
            * np.sin(np.pi * mesh.p[0, mesh.p[1] == 1.0]) ** 2
        )
    )
    assert top_velocity_error < 1.0e-12

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.0))
    draw_mesh(
        axes[0, 0],
        deformed_points["Harmonic"],
        mesh.t,
        f"Harmonic, $A={MAX_AMPLITUDE:.2f}$",
    )
    draw_mesh(
        axes[0, 1],
        deformed_points["Pseudo-elastic"],
        mesh.t,
        f"Pseudo-elastic, $A={MAX_AMPLITUDE:.2f}$",
    )

    stride = np.arange(mesh.p.shape[1])[::24]
    axes[0, 2].quiver(
        mesh.p[0, stride],
        mesh.p[1, stride],
        mesh_velocity[0, stride],
        mesh_velocity[1, stride],
        color="#d7301f",
        angles="xy",
        scale_units="xy",
        scale=4.0,
        width=0.004,
    )
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set_title("Pseudo-elastic mesh velocity at $t=0$")
    axes[0, 2].set_xlabel("$x$")
    axes[0, 2].set_ylabel("$y$")

    triangulation = mtri.Triangulation(
        deformed_points["Pseudo-elastic"][0],
        deformed_points["Pseudo-elastic"][1],
        mesh.t.T,
    )
    color_plot = axes[1, 0].tripcolor(
        triangulation,
        elastic_jacobian,
        shading="flat",
        cmap="viridis",
    )
    figure.colorbar(color_plot, ax=axes[1, 0], label="$J_m/J_{m,0}$")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("Pseudo-elastic element Jacobian ratio")
    axes[1, 0].set_xlabel("$x$")
    axes[1, 0].set_ylabel("$y$")

    colors = {"Harmonic": "#2c7fb8", "Pseudo-elastic": "#d95f0e"}
    for label, (minimum_jacobian, minimum_angle) in quality_history.items():
        axes[1, 1].plot(
            amplitudes,
            minimum_jacobian,
            "o-",
            markersize=3,
            label=label,
            color=colors[label],
        )
        axes[1, 2].plot(
            amplitudes,
            minimum_angle,
            "o-",
            markersize=3,
            label=label,
            color=colors[label],
        )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        xlabel="Interface amplitude $A$",
        ylabel="minimum $J_m/J_{m,0}$",
        title="Element inversion monitor",
    )
    axes[1, 2].set(
        xlabel="Interface amplitude $A$",
        ylabel="minimum angle [degree]",
        title="Shape-quality degradation",
    )
    axes[1, 1].legend()
    axes[1, 2].legend()
    for axis in axes[1, 1:]:
        axis.grid(alpha=0.25)

    figure.suptitle(
        "ALE mesh motion: moving top interface "
        r"$d_y=-A\sin^2(\pi x)$",
        fontsize=14,
    )
    figure.tight_layout()
    output = Path(__file__).with_name("result.png")
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print("ALE mesh-motion comparison")
    for label, (minimum_jacobian, minimum_angle) in quality_history.items():
        print(
            f"  {label:14s}: min J ratio = {minimum_jacobian[-1]:.6f}, "
            f"min angle = {minimum_angle[-1]:.3f} deg"
        )
    print(f"  boundary mesh-velocity error = {top_velocity_error:.3e}")
    print(f"  wrote {output}")


if __name__ == "__main__":
    main()
