"""Load stepping and Newton iteration for a compressible neo-Hookean unit square."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, LinearForm, MeshTri, asm
from skfem.helpers import ddot, grad


MU, LAMBDA = 1.0, 2.0


def constitutive(F):
    """First Piola stress and material tangent, arrays (..., q, element)."""
    J = np.linalg.det(np.moveaxis(F, (0, 1), (-2, -1)))
    FinvT = np.swapaxes(np.linalg.inv(np.moveaxis(F, (0, 1), (-2, -1))), -1, -2)
    FinvT = np.moveaxis(FinvT, (-2, -1), (0, 1))
    P = MU * (F - FinvT) + LAMBDA * np.log(J)[None, None, ...] * FinvT
    # A_iJkL = mu delta_ik delta_JL
    #          + (mu-lambda log J) F^-T_iL F^-T_kJ
    #          + lambda F^-T_iJ F^-T_kL
    dim = F.shape[0]
    A = np.zeros((dim, dim, dim, dim) + J.shape)
    eye = np.eye(dim)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for ell in range(dim):
                    A[i, j, k, ell] = (
                        MU * eye[i, k] * eye[j, ell]
                        + (MU - LAMBDA * np.log(J)) * FinvT[i, ell] * FinvT[k, j]
                        + LAMBDA * FinvT[i, j] * FinvT[k, ell]
                    )
    return J, P, A


@LinearForm
def residual(v, w):
    return ddot(w.P, grad(v))


@BilinearForm
def tangent(du, v, w):
    gu, gv = grad(du), grad(v)
    return np.einsum("ijkl...,kl...,ij...->...", w.A, gu, gv)


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 17), np.linspace(0.0, 1.0, 17))
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=4)
    left = basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    right_x = basis.get_dofs(lambda x: np.isclose(x[0], 1.0)).nodal["u^1"]
    fixed = np.unique(np.concatenate((left, right_x)))
    free = basis.complement_dofs(fixed)
    u = np.zeros(basis.N)
    identity = np.eye(2)[:, :, None, None]
    imposed_history, reaction_history = [], []

    def state(xvec):
        F = identity + grad(basis.interpolate(xvec))
        return constitutive(F)

    def get_residual(xvec):
        _, P, _ = state(xvec)
        return asm(residual, basis, P=P)

    for load_step, displacement in enumerate(np.linspace(0.02, 0.20, 10), 1):
        u[right_x] = displacement
        for iteration in range(1, 21):
            Jdet, P, A = state(u)
            if Jdet.min() <= 0.0:
                raise RuntimeError("element inversion in accepted state")
            r = asm(residual, basis, P=P)
            norm_r = np.linalg.norm(r[free])
            if norm_r < 1.0e-9:
                break
            K = asm(tangent, basis, A=A)
            du = np.zeros_like(u)
            du[free] = spsolve(K[free][:, free], -r[free])
            alpha = 1.0
            while alpha > 1.0e-8:
                trial = u + alpha * du
                trial_J, _, _ = state(trial)
                if trial_J.min() > 0.0 and np.linalg.norm(get_residual(trial)[free]) < norm_r:
                    u = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Newton line search failed")
        else:
            raise RuntimeError("Newton iteration did not converge")
        reaction = get_residual(u)[right_x].sum()
        imposed_history.append(displacement)
        reaction_history.append(reaction)
        print(
            f"step={load_step:2d}, ux={displacement:.3f}, Newton={iteration:2d}, "
            f"reaction={reaction:.6f}, min(detF)={state(u)[0].min():.6f}"
        )

    assert np.isfinite(u).all() and state(u)[0].min() > 0.0
    nodal_u = u[basis.nodal_dofs]
    deformed = mesh.p + nodal_u
    original_tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    deformed_tri = mtri.Triangulation(deformed[0], deformed[1], mesh.t.T)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].triplot(original_tri, color="0.75", linewidth=0.5, label="reference")
    axes[0].triplot(deformed_tri, color="tab:blue", linewidth=0.6, label="deformed")
    axes[0].set(title="Finite deformation", xlabel="x", ylabel="y", aspect="equal")
    axes[0].legend()
    axes[1].plot(imposed_history, reaction_history, "o-")
    axes[1].set(title="Load–displacement response", xlabel="prescribed $u_x$", ylabel="reaction")
    axes[1].grid(True, alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
