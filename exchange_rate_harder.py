"""Inverse Koopman beyond integrability.

Ladder of quantum Hamiltonians, increasing in non-integrability:
    Rung 1: quartic oscillator  H = p^2/2 + x^4/4
    Rung 2: Morse               H = p^2/2 + D(1 - exp(-a x))^2
    Rung 3: double well         H = p^2/2 - x^2/2 + x^4/4
    Rung 4: coupled anharmonic  H = p1^2/2 + p2^2/2 + x1^4/4 + x2^4/4 + lambda x1^2 x2^2

Stage 1 per rung: construct classical F via Recipe A (N-torus rotation).
    For any Hermitian H with eigenvalues {E_n}:
        M = T^N, F(theta_1..theta_N) = (theta_n + E_n tau)_n
        g_n(theta) = exp(-i theta_n)
        K_F is exactly diag(exp(-i E_n tau)) = U in the eigenbasis of H.
    Thus machine-precision existence is guaranteed; we verify numerically.

Stage 2 per rung: measure bandedness of U = exp(-i H tau) in the physical
position basis (1D lattice for rungs 1-3; 2D lattice for rung 4).

For rung 4 we additionally sweep coupling lambda, to test whether the locality
radius diverges with interaction strength -- a Koopman-form Bell probe.

No hbar: we set hbar=1, so U = exp(-i H tau) with tau chosen dimensionless.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.linalg as la

THIS = Path(__file__).resolve()
OUT = THIS.parent / "output_exchange_rate_harder"
OUT.mkdir(exist_ok=True, parents=True)


# =====================================================================
# Lattice builders for each rung
# =====================================================================
def H_quartic(N_lat, L=8.0):
    dx = L / N_lat
    x = (np.arange(N_lat) - N_lat / 2 + 0.5) * dx
    T = -0.5 * (-2 * np.eye(N_lat)
                + np.eye(N_lat, k=1) + np.eye(N_lat, k=-1)) / dx ** 2
    V = np.diag(0.25 * x ** 4)
    return T + V, x


def H_morse(N_lat, L=12.0, D=4.0, a=0.6):
    dx = L / N_lat
    # Morse is one-sided; center at x=0 with potential min at x=0.
    x = (np.arange(N_lat) - N_lat / 2 + 0.5) * dx
    T = -0.5 * (-2 * np.eye(N_lat)
                + np.eye(N_lat, k=1) + np.eye(N_lat, k=-1)) / dx ** 2
    V = np.diag(D * (1.0 - np.exp(-a * x)) ** 2)
    return T + V, x


def H_doublewell(N_lat, L=8.0):
    dx = L / N_lat
    x = (np.arange(N_lat) - N_lat / 2 + 0.5) * dx
    T = -0.5 * (-2 * np.eye(N_lat)
                + np.eye(N_lat, k=1) + np.eye(N_lat, k=-1)) / dx ** 2
    V = np.diag(-0.5 * x ** 2 + 0.25 * x ** 4)
    return T + V, x


def H_coupled(N1, L=6.0, lam=0.5):
    """2D: H = T_1 otimes I + I otimes T_2 + V(x1,x2).
    Dimension = N1^2. x in [-L/2, L/2] per axis.
    """
    dx = L / N1
    x = (np.arange(N1) - N1 / 2 + 0.5) * dx
    T1 = -0.5 * (-2 * np.eye(N1)
                 + np.eye(N1, k=1) + np.eye(N1, k=-1)) / dx ** 2
    I = np.eye(N1)
    T = np.kron(T1, I) + np.kron(I, T1)
    X1 = np.kron(np.diag(x), I)
    X2 = np.kron(I, np.diag(x))
    V = 0.25 * (X1 ** 4 + X2 ** 4) + lam * X1 ** 2 @ X2 ** 2
    return T + V, x


# =====================================================================
# Shared utilities
# =====================================================================
def propagator(H, tau):
    w, P = la.eigh(H)
    return P @ np.diag(np.exp(-1j * w * tau)) @ P.conj().T, w


def band_mask(N, r):
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (np.abs(i - j) <= r).astype(float)


def band_mask_2d(N1, r):
    """For a 2D lattice flattened row-major (i1 * N1 + i2), mask entries where
    both |Δi1|<=r and |Δi2|<=r (a square neighbourhood)."""
    n = N1 * N1
    i1 = np.arange(n) // N1
    i2 = np.arange(n) % N1
    di1 = np.abs(i1[:, None] - i1[None, :])
    di2 = np.abs(i2[:, None] - i2[None, :])
    return ((di1 <= r) & (di2 <= r)).astype(float)


def bandedness_curve(U, rs, mask_fn):
    Un = la.norm(U, ord="fro")
    out = []
    for r in rs:
        off = (1.0 - mask_fn(r)) * U
        rel = la.norm(off, ord="fro") / Un
        out.append({"r": int(r), "off_band_relative": float(rel)})
    return out


def locality_radius(curve, thresh=1e-3):
    """Smallest r with off_band_relative < thresh."""
    for d in curve:
        if d["off_band_relative"] < thresh:
            return d["r"]
    return None


# =====================================================================
# Stage 1 — Recipe A numerical verification for a given spectrum
# =====================================================================
def stage1_recipeA(eigvals, tau, seed=0):
    """Construct N-torus F with angles shifting by E_n * tau; sample Koopman
    matrix by evaluating g_n = exp(-i theta_n) on random torus samples.
    For any non-degenerate spectrum this must give machine-precision residual
    compared to diag(exp(-i E_n tau)).
    """
    N = len(eigvals)
    U = np.diag(np.exp(-1j * eigvals * tau))
    # random samples in the N-torus
    rng = np.random.default_rng(seed)
    n_samp = max(4 * N, 512)
    theta = rng.uniform(0, 2 * np.pi, size=(N, n_samp))
    # observables g_n on samples: G[n, s] = exp(-i theta_n_s)
    G = np.exp(-1j * theta)
    # push forward by F
    theta_next = (theta + (eigvals * tau)[:, None]) % (2 * np.pi)
    G_next = np.exp(-1j * theta_next)
    # K_F maps G -> G_next: K_F = G_next @ pinv(G)
    K_F = G_next @ np.linalg.pinv(G)
    err = la.norm(K_F - U, ord="fro")
    return {"N": int(N), "n_samp": int(n_samp),
            "KF_minus_U_fro": float(err)}


# =====================================================================
# Run all rungs
# =====================================================================
def run_rung_1d(name, H_builder, tau=0.3, N_lats=(32, 48, 64), plot_lat=48):
    print(f"\n--- Rung {name} ---")
    records = []
    for N in N_lats:
        H, x = H_builder(N)
        U, eigs = propagator(H, tau)
        s1 = stage1_recipeA(eigs, tau)
        rs = list(range(1, N))
        curve = bandedness_curve(U, rs, lambda r: band_mask(N, r))
        rad = locality_radius(curve, thresh=1e-3)
        print(f"  N_lat={N:>3}  ||K_F - U||_F={s1['KF_minus_U_fro']:.2e}  "
              f"loc radius (1e-3)={rad}")
        records.append({"N_lat": N, "stage1": s1, "curve": curve,
                        "loc_radius_1e-3": rad})
    # plot for plot_lat
    rec = next(r for r in records if r["N_lat"] == plot_lat)
    rs = [d["r"] for d in rec["curve"]]
    rels = [d["off_band_relative"] for d in rec["curve"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(rs, rels, "ko-", ms=4)
    ax.set_xlabel("bandwidth r (position basis)")
    ax.set_ylabel(r"off-band $\|U\|_F$ / $\|U\|_F$")
    ax.set_title(f"Rung {name}: bandedness of U (N_lat={plot_lat})")
    ax.axhline(1e-3, color="r", ls="--", alpha=0.5, label=r"$10^{-3}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"locality_rung_{name}.png", dpi=120)
    plt.close(fig)
    return records


def run_rung_4(tau=0.3, N1=10, lambdas=(0.0, 0.1, 0.3, 0.6, 1.0, 2.0)):
    print(f"\n--- Rung 4 (coupled anharmonic, N1={N1}) ---")
    records = []
    for lam in lambdas:
        H, x = H_coupled(N1, lam=lam)
        U, eigs = propagator(H, tau)
        s1 = stage1_recipeA(eigs, tau)
        rs = list(range(1, N1))
        curve = bandedness_curve(U, rs, lambda r: band_mask_2d(N1, r))
        rad = locality_radius(curve, thresh=1e-3)
        print(f"  lambda={lam:4.2f}  ||K_F - U||_F={s1['KF_minus_U_fro']:.2e}  "
              f"loc radius (1e-3)={rad}")
        records.append({"lambda": lam, "N1": N1, "stage1": s1,
                        "curve": curve, "loc_radius_1e-3": rad})

    # plot locality curves for all lambdas
    fig, ax = plt.subplots(figsize=(7, 5))
    for rec in records:
        rs = [d["r"] for d in rec["curve"]]
        rels = [d["off_band_relative"] for d in rec["curve"]]
        ax.semilogy(rs, rels, "o-", ms=4, label=f"λ={rec['lambda']:.2f}")
    ax.axhline(1e-3, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("2D bandwidth r (square neighbourhood, position basis)")
    ax.set_ylabel(r"off-band $\|U\|_F$ / $\|U\|_F$")
    ax.set_title(f"Rung 4: bandedness vs coupling λ (N1={N1})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "locality_rung_4.png", dpi=120)
    plt.close(fig)

    # locality radius vs lambda
    fig, ax = plt.subplots(figsize=(7, 5))
    lams = [r["lambda"] for r in records]
    rads = [r["loc_radius_1e-3"] if r["loc_radius_1e-3"] is not None else N1
            for r in records]
    ax.plot(lams, rads, "ks-", ms=6)
    ax.set_xlabel(r"coupling $\lambda$")
    ax.set_ylabel(r"locality radius $r$ for $\varepsilon(r) < 10^{-3}$")
    ax.set_title("Rung 4: locality vs interaction strength")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rung4_locality_vs_lambda.png", dpi=120)
    plt.close(fig)

    # locality radius vs N1 (fixed lambda=0.6)
    print("  --- lambda=0.6, varying N1 ---")
    N_records = []
    for n in (6, 8, 10, 12):
        H, x = H_coupled(n, lam=0.6)
        U, eigs = propagator(H, tau)
        rs = list(range(1, n))
        curve = bandedness_curve(U, rs, lambda r: band_mask_2d(n, r))
        rad = locality_radius(curve, thresh=1e-3)
        print(f"    N1={n}  loc radius (1e-3)={rad}")
        N_records.append({"N1": n, "loc_radius_1e-3": rad,
                          "curve": curve})
    fig, ax = plt.subplots(figsize=(7, 5))
    ns = [r["N1"] for r in N_records]
    rads_n = [r["loc_radius_1e-3"] if r["loc_radius_1e-3"] is not None
              else r["N1"] for r in N_records]
    ax.plot(ns, rads_n, "ks-", ms=6)
    ax.set_xlabel(r"lattice size per axis $N_1$")
    ax.set_ylabel(r"locality radius $r$ for $\varepsilon(r) < 10^{-3}$")
    ax.set_title(r"Rung 4: locality vs system size (λ=0.6)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rung4_locality_vs_N.png", dpi=120)
    plt.close(fig)
    return records, N_records


# =====================================================================
def main():
    metrics = {}

    # Rungs 1-3 (1D lattices)
    metrics["rung_1_quartic"]    = run_rung_1d("1_quartic",    H_quartic)
    metrics["rung_2_morse"]      = run_rung_1d("2_morse",      H_morse)
    metrics["rung_3_doublewell"] = run_rung_1d("3_doublewell", H_doublewell)

    # Rung 4 (2D lattice)
    rec4_lam, rec4_N = run_rung_4()
    metrics["rung_4_coupled_vs_lambda"] = rec4_lam
    metrics["rung_4_coupled_vs_N"]      = rec4_N

    # Verdict per rung
    def rung_verdict(records):
        rads = [r["loc_radius_1e-3"] for r in records]
        if all(r is not None for r in rads):
            # locality radius stable with N?
            if len(set(rads)) <= 2 and max(rads) <= 10:
                return "Pass (local, radius stable with N)"
            elif rads[-1] is not None and rads[-1] > 2 * (rads[0] or 1):
                return f"Partial (radius grows with N: {rads})"
            return f"Pass weak (radius {rads})"
        return "Fail (no r achieves 1e-3 threshold)"

    verdicts = {
        "rung_1_quartic":    rung_verdict(metrics["rung_1_quartic"]),
        "rung_2_morse":      rung_verdict(metrics["rung_2_morse"]),
        "rung_3_doublewell": rung_verdict(metrics["rung_3_doublewell"]),
    }

    # Rung 4 verdict: does loc radius grow with lambda?
    rads_vs_lam = [(r["lambda"], r["loc_radius_1e-3"]) for r in rec4_lam]
    rads_vs_N = [(r["N1"], r["loc_radius_1e-3"]) for r in rec4_N]
    verdicts["rung_4_coupled"] = {
        "stage1_always_passes": all(r["stage1"]["KF_minus_U_fro"] < 1e-8
                                    for r in rec4_lam),
        "loc_radius_vs_lambda": rads_vs_lam,
        "loc_radius_vs_N": rads_vs_N,
    }

    metrics["verdicts"] = verdicts

    with open(OUT.parent / "metrics_exchange_rate_harder.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print("\n=== EXCHANGE RATE (HARDER) SUMMARY ===")
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
