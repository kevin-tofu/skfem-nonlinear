"""Arc-length continuation of an imperfect neo-Hookean column."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, FacetBasis, LinearForm
from skfem import MeshTri, asm
from skfem.helpers import ddot, grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.nonlinear_verification import directional_derivative_errors


MU, LAME_LAMBDA = 1.0, 2.0
WIDTH = 0.10
IMPERFECTION = 0.004
ARC_LENGTH = 0.012
LOAD_SCALE = 20.0
CONTINUATION_STEPS = 70


def constitutive(F):
    moved = np.moveaxis(F, (0, 1), (-2, -1))
    determinant = np.linalg.det(moved)
    inverse_transpose = np.swapaxes(np.linalg.inv(moved), -1, -2)
    inverse_transpose = np.moveaxis(inverse_transpose, (-2, -1), (0, 1))
    stress = (
        MU * (F - inverse_transpose)
        + LAME_LAMBDA * np.log(determinant)[None, None, ...] * inverse_transpose
    )
    tangent = np.zeros((2, 2, 2, 2) + determinant.shape)
    identity = np.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for ell in range(2):
                    tangent[i, j, k, ell] = (
                        MU * identity[i, k] * identity[j, ell]
                        + (MU - LAME_LAMBDA * np.log(determinant))
                        * inverse_transpose[i, ell]
                        * inverse_transpose[k, j]
                        + LAME_LAMBDA
                        * inverse_transpose[i, j]
                        * inverse_transpose[k, ell]
                    )
    return determinant, stress, tangent


@LinearForm
def internal_force(v, w):
    return ddot(w.stress, grad(v))


@BilinearForm
def material_tangent(du, v, w):
    return np.einsum("ijkl...,kl...,ij...->...", w.tangent, grad(du), grad(v))


@LinearForm
def reference_load(v, w):
    # Integral over the top edge equals a unit downward resultant.
    return -v[1] / WIDTH


def main() -> None:
    mesh = MeshTri.init_tensor(
        np.linspace(-WIDTH / 2.0, WIDTH / 2.0, 7),
        np.linspace(0.0, 1.0, 41),
    )
    mesh.p[0] += IMPERFECTION * np.sin(np.pi * mesh.p[1])
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=4)
    top_facets = mesh.facets_satisfying(lambda x: np.isclose(x[1], 1.0))
    top_basis = FacetBasis(mesh, ElementVector(ElementTriP1()), facets=top_facets)
    bottom = basis.get_dofs(lambda x: np.isclose(x[1], 0.0)).all()
    free = basis.complement_dofs(bottom)
    external = asm(reference_load, top_basis)
    external_free = external[free]
    identity = np.eye(2)[:, :, None, None]

    def state(displacement):
        F = identity + grad(basis.interpolate(displacement))
        return constitutive(F)

    def internal(displacement):
        _, stress, _ = state(displacement)
        return asm(internal_force, basis, stress=stress)

    def tangent(displacement):
        _, _, material = state(displacement)
        return asm(material_tangent, basis, tangent=material)

    displacement = np.zeros(basis.N)
    load_factor = 0.0
    previous_increment_u = np.zeros(free.size)
    previous_increment_load = 1.0
    load_history = [0.0]
    vertical_history = [0.0]
    lateral_history = [0.0]
    iteration_history = []
    saved_states = [(0.0, displacement.copy())]
    top_y = basis.get_dofs(lambda x: np.isclose(x[1], 1.0)).nodal["u^2"]

    for continuation_step in range(1, CONTINUATION_STEPS + 1):
        base_u = displacement.copy()
        base_load = load_factor
        K = tangent(base_u)[free][:, free]
        load_direction = spsolve(K, external_free)
        orientation = (
            load_direction @ previous_increment_u
            + LOAD_SCALE**2 * previous_increment_load
        )
        sign = 1.0 if orientation >= 0.0 else -1.0
        normalization = np.sqrt(
            load_direction @ load_direction + LOAD_SCALE**2
        )
        increment_load = sign * ARC_LENGTH / normalization
        increment_u = increment_load * load_direction
        displacement[free] = base_u[free] + increment_u
        load_factor = base_load + increment_load

        for iteration in range(1, 21):
            total_increment_u = displacement[free] - base_u[free]
            total_increment_load = load_factor - base_load
            residual = internal(displacement)[free] - load_factor * external_free
            constraint = (
                total_increment_u @ total_increment_u
                + LOAD_SCALE**2 * total_increment_load**2
                - ARC_LENGTH**2
            )
            combined_norm = np.sqrt(
                residual @ residual + (constraint / ARC_LENGTH) ** 2
            )
            if combined_norm < 1.0e-9:
                break
            K = tangent(displacement)[free][:, free]
            last_row = csr_matrix(
                np.concatenate(
                    (
                        2.0 * total_increment_u,
                        [2.0 * LOAD_SCALE**2 * total_increment_load],
                    )
                )[None, :]
            )
            bordered = bmat(
                [
                    [K, csr_matrix((-external_free)[:, None])],
                    [last_row[:, :-1], last_row[:, -1:]],
                ],
                format="csr",
            )
            correction = spsolve(
                bordered,
                -np.concatenate((residual, [constraint])),
            )
            displacement[free] += correction[:-1]
            load_factor += correction[-1]
            if state(displacement)[0].min() <= 0.0:
                raise RuntimeError("element inversion during arc-length correction")
        else:
            raise RuntimeError(f"arc-length correction failed at step {continuation_step}")

        previous_increment_u = displacement[free] - base_u[free]
        previous_increment_load = load_factor - base_load
        nodal_u = displacement[basis.nodal_dofs]
        load_history.append(load_factor)
        vertical_history.append(-np.mean(displacement[top_y]))
        lateral_history.append(np.max(np.abs(nodal_u[0])))
        iteration_history.append(iteration)
        if continuation_step in (20, 40, 55, 70):
            saved_states.append((load_factor, displacement.copy()))
        if continuation_step % 10 == 0:
            print(
                f"step={continuation_step:2d}, load={load_factor:+.6e}, "
                f"top shortening={vertical_history[-1]:.5f}, "
                f"lateral={lateral_history[-1]:.5f}, Newton={iteration:2d}"
            )

    tangent_steps, tangent_errors = directional_derivative_errors(
        internal,
        tangent,
        displacement,
        free,
    )
    print(
        f"maximum tracked load={max(load_history):.6e}, final load={load_factor:.6e}, "
        f"minimum tangent error={tangent_errors.min():.3e}"
    )
    assert np.isfinite(displacement).all()
    assert tangent_errors.min() < 1.0e-7
    assert max(lateral_history) > 2.0 * IMPERFECTION

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for factor, saved_u in saved_states:
        nodal_u = saved_u[basis.nodal_dofs]
        deformed = mesh.p + nodal_u
        deformed[0] = mesh.p[0] + 3.0 * nodal_u[0]
        triangulation = mtri.Triangulation(deformed[0], deformed[1], mesh.t.T)
        axes[0, 0].triplot(
            triangulation,
            linewidth=0.45,
            label=f"λ={factor:.4g}",
        )
    axes[0, 0].set(title="Post-buckling configurations ($u_x\\times3$)", xlabel="display x", ylabel="y")
    axes[0, 0].legend(fontsize=8, loc="upper right")

    axes[0, 1].plot(vertical_history, load_history, "o-", markersize=3)
    axes[0, 1].set(title="Arc-length equilibrium path", xlabel="top shortening", ylabel="load factor")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(lateral_history, load_history, "o-", markersize=3, label="equilibrium path")
    iteration_axis = axes[1, 0].twinx()
    iteration_axis.step(
        lateral_history[1:],
        iteration_history,
        where="mid",
        color="tab:orange",
        label="Newton iterations",
    )
    axes[1, 0].set(title="Buckling amplitude", xlabel="maximum lateral displacement", ylabel="load factor")
    iteration_axis.set_ylabel("corrector iterations")
    axes[1, 0].grid(True, alpha=0.3)
    lines = axes[1, 0].lines + iteration_axis.lines
    axes[1, 0].legend(lines, [line.get_label() for line in lines], loc="best")

    axes[1, 1].loglog(tangent_steps, tangent_errors, "o-", label="hyperelastic tangent")
    axes[1, 1].loglog(
        tangent_steps,
        tangent_errors[0] * (tangent_steps / tangent_steps[0]) ** 2,
        "k--",
        label=r"$O(h^2)$",
    )
    axes[1, 1].invert_xaxis()
    axes[1, 1].set(title="Material-tangent verification", xlabel="difference step", ylabel="relative error")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
