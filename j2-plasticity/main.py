"""Small-strain plane-strain J2 plasticity with isotropic hardening.

The unit square is pulled and then unloaded by prescribed horizontal
displacement.  Plastic strain and accumulated plastic strain are stored at
quadrature points and committed only after global Newton convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, LinearForm, MeshTri, asm
from skfem.helpers import ddot, grad, sym_grad


YOUNG, POISSON = 1000.0, 0.30
YIELD_STRESS, HARDENING = 2.0, 20.0
MU = YOUNG / (2.0 * (1.0 + POISSON))
BULK = YOUNG / (3.0 * (1.0 - 2.0 * POISSON))
I3 = np.eye(3)[:, :, None, None]


def material_update(strain, plastic_strain_old, alpha_old):
    """Three-dimensional radial return evaluated at plane-strain states."""
    elastic_strain = strain - plastic_strain_old
    trace = np.einsum("ii...->...", elastic_strain)
    deviatoric = elastic_strain - trace[None, None, ...] * I3 / 3.0
    trial_stress = 2.0 * MU * deviatoric + BULK * trace[None, None, ...] * I3
    mean_stress = np.einsum("ii...->...", trial_stress) / 3.0
    trial_s = trial_stress - mean_stress[None, None, ...] * I3
    q_trial = np.sqrt(1.5 * np.einsum("ij...,ij...->...", trial_s, trial_s))
    yield_function = q_trial - (YIELD_STRESS + HARDENING * alpha_old)
    plastic = yield_function > 0.0

    dgamma = np.where(plastic, yield_function / (3.0 * MU + HARDENING), 0.0)
    safe_q = np.maximum(q_trial, np.finfo(float).eps)
    flow_direction = 1.5 * trial_s / safe_q[None, None, ...]
    stress = trial_stress - 2.0 * MU * dgamma[None, None, ...] * flow_direction
    plastic_strain = plastic_strain_old + dgamma[None, None, ...] * flow_direction
    alpha = alpha_old + dgamma
    return stress, plastic_strain, alpha, plastic


def stress_and_numerical_tangent(strain, plastic_strain_old, alpha_old):
    """Return stress and a consistent local algorithmic tangent.

    Differentiating the complete return-mapping algorithm numerically keeps
    this teaching example compact while retaining Newton consistency.
    """
    stress, plastic_strain, alpha, plastic = material_update(
        strain, plastic_strain_old, alpha_old
    )
    tangent = np.zeros((3, 3, 3, 3) + strain.shape[2:])
    step = 1.0e-7
    for k in range(3):
        for ell in range(k, 3):
            direction = np.zeros_like(strain)
            direction[k, ell] = 1.0 if k == ell else 0.5
            direction[ell, k] = 1.0 if k == ell else 0.5
            plus = material_update(
                strain + step * direction, plastic_strain_old, alpha_old
            )[0]
            minus = material_update(
                strain - step * direction, plastic_strain_old, alpha_old
            )[0]
            derivative = (plus - minus) / (2.0 * step)
            tangent[:, :, k, ell] = derivative
            tangent[:, :, ell, k] = derivative
    return stress, tangent, plastic_strain, alpha, plastic


@LinearForm
def internal_force(v, w):
    return ddot(w.stress[:2, :2], sym_grad(v))


@BilinearForm
def tangent_form(du, v, w):
    eu, ev = sym_grad(du), sym_grad(v)
    return np.einsum("ijkl...,kl...,ij...->...", w.C[:2, :2, :2, :2], eu, ev)


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 17), np.linspace(0.0, 1.0, 17))
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=2)
    left = basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    right_x = basis.get_dofs(lambda x: np.isclose(x[0], 1.0)).nodal["u^1"]
    fixed = np.unique(np.concatenate((left, right_x)))
    free = basis.complement_dofs(fixed)

    quadrature_shape = basis.global_coordinates().shape[1:]
    plastic_strain = np.zeros((3, 3) + quadrature_shape)
    alpha = np.zeros(quadrature_shape)
    u = np.zeros(basis.N)
    displacement_history, reaction_history = [], []
    # Loading beyond first yield, followed by complete displacement unloading.
    load_path = np.concatenate((np.linspace(0.001, 0.010, 10), np.linspace(0.009, 0.0, 10)))

    def total_strain(xvec):
        strain2 = sym_grad(basis.interpolate(xvec))
        strain3 = np.zeros((3, 3) + strain2.shape[2:])
        strain3[:2, :2] = strain2
        return strain3

    def response(xvec):
        return stress_and_numerical_tangent(total_strain(xvec), plastic_strain, alpha)

    for step_number, prescribed_u in enumerate(load_path, 1):
        u[right_x] = prescribed_u
        for iteration in range(1, 31):
            stress, C, trial_plastic_strain, trial_alpha, yielded = response(u)
            residual = asm(internal_force, basis, stress=stress)
            residual_norm = np.linalg.norm(residual[free])
            if residual_norm < 1.0e-9:
                break
            K = asm(tangent_form, basis, C=C)
            increment = np.zeros_like(u)
            increment[free] = spsolve(K[free][:, free], -residual[free])
            alpha_ls = 1.0
            while alpha_ls > 1.0e-8:
                trial_u = u + alpha_ls * increment
                trial_stress = response(trial_u)[0]
                trial_residual = asm(internal_force, basis, stress=trial_stress)
                if np.linalg.norm(trial_residual[free]) < residual_norm:
                    u = trial_u
                    break
                alpha_ls *= 0.5
            else:
                raise RuntimeError(f"line search failed at load step {step_number}")
        else:
            raise RuntimeError(f"Newton failed at load step {step_number}")

        # Commit history exactly once, after equilibrium has converged.
        stress, _, plastic_strain, alpha, yielded = response(u)
        reaction = asm(internal_force, basis, stress=stress)[right_x].sum()
        displacement_history.append(prescribed_u)
        reaction_history.append(reaction)
        print(
            f"step={step_number:2d}, ux={prescribed_u:7.4f}, "
            f"reaction={reaction:8.4f}, Newton={iteration:2d}, "
            f"plastic_qp={np.count_nonzero(yielded):4d}, max(alpha)={alpha.max():.5f}"
        )

    residual_displacement = np.linalg.norm(u)
    print(
        f"after unloading: ||u||={residual_displacement:.4e}, "
        f"max(alpha)={alpha.max():.4e}"
    )
    assert np.isfinite(u).all() and alpha.max() > 0.0

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    element_alpha = alpha.mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].plot(displacement_history, reaction_history, "o-", label="reaction")
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].set(title="Elastoplastic hysteresis", xlabel="prescribed $u_x$", ylabel="reaction")
    axes[0].grid(True, alpha=0.3)
    image = axes[1].tripcolor(triangulation, facecolors=element_alpha, shading="flat", cmap="inferno")
    fig.colorbar(image, ax=axes[1], label=r"accumulated plastic strain $\alpha$")
    axes[1].set(title="Plastic strain after unloading", xlabel="x", ylabel="y", aspect="equal")
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
