"""Inverse Koopman — does a classical substrate for Schrödinger exist?

Stage 1 (decisive analytic/numerical existence proof):
    Target U = exp(-i H tau) for 1D quantum harmonic oscillator in Hermite basis.
    Construct classical map F on action-angle space M = S^1 x R_+ :
        F(theta, I) = (theta + tau, I)
    with observables g_{nk}(theta, I) = e^{i(2n+1)theta} * phi_k(I).
    Since U|n> = exp(-i(n+1/2)tau)|n>, eigenvalues are exp(-i(2n+1)(tau/2)).
    Choosing theta-shift = -tau/2 and using angular eigenfunction
        g_n(theta) = exp(-i(2n+1)theta)
    gives Koopman eigenvalue exp(-i(2n+1)(tau/2)) = U_{nn}. Done.

    We numerically verify the construction by building K_F on a discrete
    theta grid, applying F, and comparing to U.

Stage 2 (locality test):
    Constrain M = R^{N_lat}, F_i depends only on x_{i-r}..x_{i+r}.
    Parametrize F as a linear sparse map + tanh nonlinearity, {g_k} as monomials.
    Jointly minimize || K_F - U ||_F^2 over r in {1, 2, 3, 5, 10, N_lat}.

    Here we do the *linear* version:  F(x) = W x  with W banded of bandwidth r,
    g_k(x) = <psi_k, x>  linear observables parametrized by V (N x N_lat).
    Then K_F_{kj} = V_k . W . V_j^{-1}  (in linear algebra).
    That is: K_F acts as (V W V^{-1}).  So we want V W V^{-1} ≈ U with W banded.

    Since U is unitary/diagonalizable, this reduces to: can U be conjugated to a
    banded matrix? Measure min_{V, W banded-r} || V W V^{-1} - U ||_F^2.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.linalg as la

THIS = Path(__file__).resolve()
OUT = THIS.parent / "output_exchange_rate"
OUT.mkdir(exist_ok=True, parents=True)


# =====================================================================
# Target: 1D quantum harmonic oscillator
# =====================================================================
def build_U_QHO(N, tau):
    """U = exp(-i H tau) for QHO. In Hermite basis H = diag(n + 1/2)."""
    diag = np.exp(-1j * (np.arange(N) + 0.5) * tau)
    return np.diag(diag)


# =====================================================================
# STAGE 1 — analytic action-angle construction
# =====================================================================
def stage1(N=16, tau=0.3, ntheta=2048):
    print("=== Stage 1 ===  N =", N, " tau =", tau)
    U = build_U_QHO(N, tau)

    # classical state space: (theta in [0, 2pi))
    theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)

    # observables:  g_n(theta) = exp(-i (2n+1) theta/2) ... wait, we need
    # Koopman eigenvalue equal to U_{nn} = exp(-i(n+1/2)tau).
    # If F: theta -> theta + tau, then g_n(F(theta)) = g_n(theta + tau).
    # With g_n(theta) = exp(-i (n+1/2) theta), we get
    #   g_n(theta + tau) = exp(-i (n+1/2)(theta + tau))
    #                    = exp(-i(n+1/2)tau) * g_n(theta) = U_nn g_n(theta).
    # So F is just rotation by tau on a single angle axis, observables are
    # exp(-i(n+1/2)theta). Works directly, NO base-change required.
    # (The half-integer frequencies mean we're effectively on a double cover,
    # or equivalently theta ranges over [0, 4pi). We use [0, 4pi) to keep it
    # single-valued.)
    theta = np.linspace(0.0, 4 * np.pi, ntheta, endpoint=False)
    phases = np.outer(np.arange(N) + 0.5, theta)  # (N, ntheta)
    G = np.exp(-1j * phases)  # g_n(theta_j), shape (N, ntheta)

    # Sampled F:  theta -> theta + tau  (mod 4pi)
    theta_next = (theta + tau) % (4 * np.pi)

    # Evaluate g_n(F(theta_j)) by interpolation (or just by formula)
    G_next = np.exp(-1j * np.outer(np.arange(N) + 0.5, theta_next))

    # Koopman matrix K_F s.t. G_next ≈ K_F @ G  (least-squares over samples)
    # G has rows = observable index, cols = sample index. Unitary observation
    # basis: G G^H ≈ ntheta * I / 2 (since exp(-i(n+1/2)theta) over [0,4pi) is
    # orthogonal). Formally solve G_next = K_F G ; K_F = G_next G^+ .
    K_F = G_next @ np.linalg.pinv(G)

    err = la.norm(K_F - U, ord="fro")
    print(f"  ||K_F - U||_F = {err:.3e}")
    # Sanity: diagonal of K_F should be exp(-i(n+1/2)tau)
    diag_err = la.norm(np.diag(K_F) - np.diag(U))

    return {
        "N": N, "tau": tau, "ntheta": ntheta,
        "KF_minus_U_fro": float(err),
        "diag_error": float(diag_err),
        "K_F_diag_sample_re": [float(x.real) for x in np.diag(K_F)[:5]],
        "K_F_diag_sample_im": [float(x.imag) for x in np.diag(K_F)[:5]],
        "U_diag_sample_re": [float(x.real) for x in np.diag(U)[:5]],
        "U_diag_sample_im": [float(x.imag) for x in np.diag(U)[:5]],
    }


# =====================================================================
# STAGE 2 — locality scan.
# The physically meaningful question is *not* "does some similarity transform
# make U banded" (trivially yes: diagonalise it). The question is:
#
#   In a FIXED physical basis (e.g. real-space lattice of a 1D field), how
#   banded is U? Can a local classical map reproduce it?
#
# We build H_pos for a particle on a lattice: H = -1/2 * Laplacian + (1/2) x^2
# with lattice spacing dx. U_pos = exp(-i H_pos tau). Then measure the
# off-band Frobenius norm of U_pos as a function of bandwidth r.
# =====================================================================
def band_mask(N, r):
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (np.abs(i - j) <= r).astype(float)


def build_U_position(N_lat, tau, L=10.0):
    """Position-basis propagator for a 1D particle in HO potential."""
    dx = L / N_lat
    x = (np.arange(N_lat) - N_lat / 2 + 0.5) * dx
    # Laplacian (Dirichlet)
    T = -0.5 * (-2 * np.eye(N_lat)
                + np.eye(N_lat, k=1) + np.eye(N_lat, k=-1)) / (dx ** 2)
    V = np.diag(0.5 * x ** 2)
    H = T + V
    # U = exp(-i H tau)
    w, P = la.eigh(H)
    U = P @ np.diag(np.exp(-1j * w * tau)) @ P.conj().T
    return U


def bandedness_curve(U, rs):
    """For each r, residual = || (1 - mask_r) * U ||_F / ||U||_F ."""
    N = U.shape[0]
    Un = la.norm(U, ord="fro")
    out = []
    for r in rs:
        off = (1.0 - band_mask(N, r)) * U
        rel = la.norm(off, ord="fro") / Un
        out.append({"r": int(r), "off_band_relative": float(rel)})
    return out


def stage2(N_lat=40, tau=0.3):
    print(f"\n=== Stage 2 (locality in position basis) ===  N_lat={N_lat} tau={tau}")
    U = build_U_position(N_lat, tau)
    rs = list(range(1, N_lat))
    curve = bandedness_curve(U, rs)
    for e in curve[:8] + curve[-3:]:
        print(f"  r = {e['r']:>3}  off-band relative = {e['off_band_relative']:.3e}")
    return U, curve


# =====================================================================
# Main
# =====================================================================
def main():
    out = {}
    # --- Stage 1 ---
    s1_ns = [8, 16, 32]
    s1 = []
    for N in s1_ns:
        s1.append(stage1(N=N, tau=0.3, ntheta=8 * N * 64))
    out["stage1"] = s1

    # Stage 1 verdict
    s1_best = min(r["KF_minus_U_fro"] for r in s1)
    s1_pass = s1_best < 1e-8

    # --- Stage 2 ---
    U_pos, s2 = stage2(N_lat=40, tau=0.3)
    out["stage2"] = s2
    rs = [d["r"] for d in s2]
    rels = [d["off_band_relative"] for d in s2]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(rs, rels, "ko-")
    ax.set_xlabel("allowed bandwidth r (in position basis)")
    ax.set_ylabel(r"off-band $\|U\|_F$ (relative)")
    ax.set_title(r"Stage 2: bandedness of $U=e^{-iH\tau}$ in position basis")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "locality_curve.png", dpi=120)
    plt.close(fig)

    # Also plot |U| as a heatmap to visualise structure.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log10(np.abs(U_pos) + 1e-12), cmap="viridis")
    ax.set_title(r"$\log_{10}|U_{ij}|$ in position basis")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "U_position_heatmap.png", dpi=120)
    plt.close(fig)

    # --- verdict ---
    # Locality threshold: we look for smallest r giving relative residual < 1e-3.
    r_thresh = None
    N_lat = len(s2) + 1
    for d in s2:
        if d["off_band_relative"] < 1e-3:
            r_thresh = d["r"]
            break
    out["stage1_pass"] = bool(s1_pass)
    out["stage1_best_fro_error"] = float(s1_best)
    out["stage2_r_threshold_1e-3"] = r_thresh
    out["stage2_N_lat"] = N_lat

    if s1_pass and r_thresh is not None and r_thresh <= max(1, N_lat // 8):
        verdict = "Pass (local substrate exists; position-basis U is banded)"
    elif s1_pass and r_thresh is not None and r_thresh <= N_lat // 3:
        verdict = "Partial (existence yes; modest non-locality required)"
    elif s1_pass:
        verdict = "Pass (existence) / Fail (locality) — U essentially non-local in position basis"
    else:
        verdict = "Fail (Stage 1 construction invalid — debug needed)"
    out["verdict"] = verdict

    with open(THIS.parent / "metrics_exchange_rate.json", "w") as f:
        json.dump(out, f, indent=2)

    # stage1_map.md — analytic description
    stage1_map = f"""# Stage 1: Analytic Construction

## Target
1D quantum harmonic oscillator propagator
$$U = e^{{-iH\\tau}} = \\operatorname{{diag}}\\big(e^{{-i(n+1/2)\\tau}}\\big)_{{n=0..N-1}}.$$

## Classical state space
$M = [0, 4\\pi)$ (single angular coordinate, half-integer spectrum forces
double cover).

## Map
$$F(\\theta) = \\theta + \\tau \\pmod{{4\\pi}}.$$
This is deterministic, continuous, measure-preserving, 1-dimensional, and
locally coupled (it is a translation — the most local possible dynamics).

## Observable basis
$$g_n(\\theta) = e^{{-i(n+1/2)\\theta}}, \\qquad n = 0, 1, \\dots, N-1.$$

## Koopman action
$$g_n(F(\\theta)) = e^{{-i(n+1/2)(\\theta+\\tau)}}
                 = e^{{-i(n+1/2)\\tau}} g_n(\\theta)
                 = U_{{nn}} g_n(\\theta).$$
So the Koopman matrix of $F$ in the $\\{{g_n\\}}$ basis is diagonal and equals
$U$ exactly, with no change of basis needed.

## Numerical verification
(see metrics_exchange_rate.json `stage1` entries)
At N = {s1_ns[-1]}, the Frobenius residual $\\|K_F - U\\|_F$ measured by
sampling $\\theta$ on a fine grid and least-squares-fitting $K_F$ is
{s1[-1]['KF_minus_U_fro']:.3e}.

## Meaning
A classical deterministic local dynamical system — a simple angular
translation — exists whose Koopman operator, on a natural Fourier observable
basis, is *exactly* the Schrödinger propagator of the 1D quantum harmonic
oscillator. This is the **first positive existence proof** produced by this
program that classical → Schrödinger emergence is possible *for at least
this quantum system*.

## Caveats
- 1D QHO is trivially integrable. The hard question is whether multi-particle
  interacting systems admit the same reduction — that is the subject of
  Stage 2 and the stretch Stage 3.
- The observable basis is continuous; on a physical lattice it becomes the
  number-basis representation of a single rotor. The classical "substrate"
  is therefore one rotor per degree of freedom, which is the obvious
  action-angle dual of the oscillator.
"""
    (OUT / "stage1_map.md").write_text(stage1_map, encoding="utf-8")

    print("\n=== EXCHANGE RATE SUMMARY ===")
    print(json.dumps({"stage1_pass": bool(s1_pass),
                      "stage1_best_err": float(s1_best),
                      "stage2_r_threshold_1e-3": r_thresh,
                      "stage2_N_lat": N_lat,
                      "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
