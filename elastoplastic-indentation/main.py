"""Plane-strain J2 elastoplastic body indented by a frictionless rigid punch."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, LinearForm, MeshTri, asm
from skfem.helpers import ddot, sym_grad


YOUNG, POISSON = 1000.0, 0.30
YIELD_STRESS, HARDENING = 20.0, 100.0
MU = YOUNG / (2.0 * (1.0 + POISSON))
BULK = YOUNG / (3.0 * (1.0 - 2.0 * POISSON))
CONTACT_PENALTY = 2.0e5
I3 = np.eye(3)[:, :, None, None]


def indenter_initial_gap(x):
    return 0.002 + 0.8 * (x - 0.5) ** 2


def material_update(strain, plastic_strain_old, alpha_old):
    elastic_strain = strain - plastic_strain_old
    trace_strain = np.einsum("ii...->...", elastic_strain)
    deviatoric_strain = elastic_strain - trace_strain[None, None, ...] * I3 / 3.0
    trial_stress = (
        2.0 * MU * deviatoric_strain
        + BULK * trace_strain[None, None, ...] * I3
    )
    mean_stress = np.einsum("ii...->...", trial_stress) / 3.0
    trial_deviator = trial_stress - mean_stress[None, None, ...] * I3
    equivalent_trial = np.sqrt(
        1.5 * np.einsum("ij...,ij...->...", trial_deviator, trial_deviator)
    )
    yield_function = equivalent_trial - (YIELD_STRESS + HARDENING * alpha_old)
    plastic = yield_function > 0.0
    plastic_increment = np.where(
        plastic, yield_function / (3.0 * MU + HARDENING), 0.0
    )
    safe_equivalent = np.maximum(equivalent_trial, np.finfo(float).eps)
    flow = 1.5 * trial_deviator / safe_equivalent[None, None, ...]
    stress = trial_stress - 2.0 * MU * plastic_increment[None, None, ...] * flow
    plastic_strain = plastic_strain_old + plastic_increment[None, None, ...] * flow
    alpha = alpha_old + plastic_increment
    return stress, plastic_strain, alpha, plastic


def response_with_tangent(strain, plastic_strain_old, alpha_old):
    stress, plastic_strain, alpha, plastic = material_update(
        strain, plastic_strain_old, alpha_old
    )
    tangent = np.zeros((3, 3, 3, 3) + strain.shape[2:])
    difference_step = 1.0e-7
    for k in range(3):
        for ell in range(k, 3):
            direction = np.zeros_like(strain)
            direction[k, ell] = 1.0 if k == ell else 0.5
            direction[ell, k] = 1.0 if k == ell else 0.5
            plus = material_update(
                strain + difference_step * direction,
                plastic_strain_old,
                alpha_old,
            )[0]
            minus = material_update(
                strain - difference_step * direction,
                plastic_strain_old,
                alpha_old,
            )[0]
            derivative = (plus - minus) / (2.0 * difference_step)
            tangent[:, :, k, ell] = derivative
            tangent[:, :, ell, k] = derivative
    return stress, tangent, plastic_strain, alpha, plastic


@LinearForm
def internal_force(v, w):
    return ddot(w.stress[:2, :2], sym_grad(v))


@BilinearForm
def material_tangent(du, v, w):
    return np.einsum(
        "ijkl...,kl...,ij...->...",
        w.tangent[:2, :2, :2, :2],
        sym_grad(du),
        sym_grad(v),
    )


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 33), np.linspace(0.0, 0.5, 17))
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=2)
    bottom = basis.get_dofs(lambda x: np.isclose(x[1], 0.0)).all()
    top = basis.get_dofs(lambda x: np.isclose(x[1], 0.5))
    top_y = top.nodal["u^2"]
    free = basis.complement_dofs(bottom)
    top_x_coordinates = basis.doflocs[0, top_y]
    initial_gaps = indenter_initial_gap(top_x_coordinates)

    quadrature_shape = basis.global_coordinates().shape[1:]
    plastic_strain = np.zeros((3, 3) + quadrature_shape)
    alpha = np.zeros(quadrature_shape)
    displacement = np.zeros(basis.N)
    loading = np.linspace(0.0, 0.08, 33)
    unloading = np.linspace(0.0775, 0.0, 32)
    depth_path = np.concatenate((loading, unloading))
    force_history = []
    contact_width_history = []
    plastic_history = []
    iteration_history = []
    peak_state = None

    def total_strain(state):
        strain2 = sym_grad(basis.interpolate(state))
        strain3 = np.zeros((3, 3) + strain2.shape[2:])
        strain3[:2, :2] = strain2
        return strain3

    def contact(state, depth):
        gap = initial_gaps - depth - state[top_y]
        active = gap < 0.0
        force = np.where(active, -CONTACT_PENALTY * gap, 0.0)
        return gap, active, force

    def trial_response(state):
        return response_with_tangent(total_strain(state), plastic_strain, alpha)

    for step, depth in enumerate(depth_path, 1):
        for iteration in range(1, 41):
            (
                stress,
                tangent,
                trial_plastic_strain,
                trial_alpha,
                yielded,
            ) = trial_response(displacement)
            gap, active, contact_force = contact(displacement, depth)
            residual = asm(internal_force, basis, stress=stress)
            residual[top_y] += contact_force
            norm_residual = np.linalg.norm(residual[free])
            if norm_residual < 1.0e-8:
                break
            system_tangent = asm(material_tangent, basis, tangent=tangent)
            contact_diagonal = np.zeros(basis.N)
            contact_diagonal[top_y[active]] = CONTACT_PENALTY
            system_tangent += diags(contact_diagonal)
            increment = np.zeros_like(displacement)
            increment[free] = spsolve(
                system_tangent[free][:, free], -residual[free]
            )

            line_alpha = 1.0
            while line_alpha > 1.0e-8:
                trial_displacement = displacement + line_alpha * increment
                trial_stress = trial_response(trial_displacement)[0]
                trial_contact = contact(trial_displacement, depth)[2]
                trial_residual = asm(
                    internal_force, basis, stress=trial_stress
                )
                trial_residual[top_y] += trial_contact
                if np.linalg.norm(trial_residual[free]) < norm_residual:
                    displacement = trial_displacement
                    break
                line_alpha *= 0.5
            else:
                raise RuntimeError(f"line search failed at indentation step {step}")
        else:
            raise RuntimeError(f"indentation Newton failed at step {step}")

        stress, _, plastic_strain, alpha, yielded = trial_response(displacement)
        gap, active, contact_force = contact(displacement, depth)
        contact_width = (
            top_x_coordinates[active].max() - top_x_coordinates[active].min()
            if np.count_nonzero(active) > 1
            else 0.0
        )
        force_history.append(contact_force.sum())
        contact_width_history.append(contact_width)
        plastic_history.append(alpha.max())
        iteration_history.append(iteration)
        if np.isclose(depth, loading[-1]):
            peak_state = (
                displacement.copy(),
                alpha.copy(),
                contact_force.copy(),
                active.copy(),
            )
        print(
            f"step={step:2d}, depth={depth:.4f}, force={contact_force.sum():.4f}, "
            f"contact={np.count_nonzero(active):2d}, plastic_qp={np.count_nonzero(yielded):4d}, "
            f"max(alpha)={alpha.max():.4f}, Newton={iteration:2d}"
        )

    assert peak_state is not None
    assert alpha.max() > 0.0
    assert force_history[-1] < 1.0e-6
    residual_imprint = -np.min(displacement[top_y])
    print(
        f"after unloading: residual imprint={residual_imprint:.6e}, "
        f"max(alpha)={alpha.max():.6e}"
    )

    peak_displacement, peak_alpha, peak_force, peak_active = peak_state
    nodal_displacement = peak_displacement[basis.nodal_dofs]
    deformed = mesh.p + nodal_displacement
    triangulation = mtri.Triangulation(deformed[0], deformed[1], mesh.t.T)
    element_alpha = peak_alpha.mean(axis=1)
    order = np.argsort(top_x_coordinates)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].triplot(triangulation, color="tab:blue", linewidth=0.5)
    indenter_surface = 0.5 + initial_gaps - loading[-1]
    axes[0, 0].plot(
        top_x_coordinates[order],
        indenter_surface[order],
        "k-",
        linewidth=2,
        label="rigid indenter",
    )
    axes[0, 0].set(title="Peak indentation state", xlabel="x", ylabel="y", aspect="equal")
    axes[0, 0].legend()

    plastic_image = axes[0, 1].tripcolor(
        triangulation, facecolors=element_alpha, shading="flat", cmap="inferno"
    )
    fig.colorbar(plastic_image, ax=axes[0, 1], label="accumulated plastic strain")
    axes[0, 1].set(title="Plastic zone at peak load", xlabel="x", ylabel="y", aspect="equal")

    axes[0, 2].plot(top_x_coordinates, peak_force, "o-")
    axes[0, 2].set(title="Peak contact-force distribution", xlabel="x", ylabel="nodal contact force")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(depth_path, force_history, "o-", markersize=3)
    axes[1, 0].set(title="Elastoplastic indentation hysteresis", xlabel="indentation depth", ylabel="contact force")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(depth_path, plastic_history, label=r"$\max\alpha$")
    width_axis = axes[1, 1].twinx()
    width_axis.plot(
        depth_path,
        contact_width_history,
        color="tab:orange",
        label="contact width",
    )
    axes[1, 1].set(title="Plasticity and contact growth", xlabel="indentation depth", ylabel="maximum plastic strain")
    width_axis.set_ylabel("contact width")
    axes[1, 1].grid(True, alpha=0.3)
    lines = axes[1, 1].lines + width_axis.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], loc="best")

    axes[1, 2].step(range(1, len(depth_path) + 1), iteration_history, where="mid")
    axes[1, 2].set(title="Coupled material/contact Newton work", xlabel="load step", ylabel="iterations")
    axes[1, 2].grid(True, alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
