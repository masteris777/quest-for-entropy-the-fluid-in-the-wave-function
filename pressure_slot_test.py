from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class PacketConfig:
    label: str
    amplitude: float
    sigma: float
    kx: float


@dataclass(frozen=True)
class SimulationConfig:
    nx: int = 256
    dx: float = 1.0
    dt: float = 0.08
    steps: int = 1800
    sample_stride: int = 8
    c: float = 1.0
    alpha: float = 0.82
    gamma: float = 0.003
    background_k: float = 0.05
    packet_center: float = 48.0
    coarse_sigmas: tuple[float, ...] = (1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0)
    mask_fraction: float = 0.10
    report_residual_threshold: float = 0.35
    omega_ref: float = 0.72


def default_packets() -> list[PacketConfig]:
    return [
        PacketConfig(label="base", amplitude=1.00, sigma=10.0, kx=0.55),
        PacketConfig(label="narrow_fast", amplitude=0.86, sigma=7.2, kx=0.72),
        PacketConfig(label="wide_slow", amplitude=1.12, sigma=13.0, kx=0.42),
    ]


def laplacian_1d(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1) - 2.0 * field + np.roll(field, 1)) / (dx * dx)


def total_force(u: np.ndarray, v: np.ndarray, config: SimulationConfig) -> np.ndarray:
    return (
        config.c**2 * laplacian_1d(u, config.dx)
        - config.background_k * u
        - config.alpha * np.sin(u)
        - config.gamma * v
    )


def initialize_packet(config: SimulationConfig, packet: PacketConfig) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.arange(config.nx, dtype=np.float64) * config.dx
    envelope = np.exp(-0.5 * ((x - config.packet_center) / packet.sigma) ** 2)
    phase = packet.kx * (x - config.packet_center)
    omega = np.sqrt(config.background_k + config.alpha + config.c**2 * packet.kx**2)
    u0 = packet.amplitude * envelope * np.cos(phase)
    v0 = packet.amplitude * omega * envelope * np.sin(phase)
    return u0, v0, float(omega)


def step_system(u: np.ndarray, v: np.ndarray, config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    force_0 = total_force(u, v, config)
    v_half = v + 0.5 * config.dt * force_0
    u_next = u + config.dt * v_half
    force_1 = total_force(u_next, v_half, config)
    v_next = v_half + 0.5 * config.dt * force_1
    return u_next, v_next


def run_packet(config: SimulationConfig, packet: PacketConfig) -> dict[str, object]:
    u, v, omega = initialize_packet(config, packet)
    samples_u: list[np.ndarray] = []
    samples_v: list[np.ndarray] = []
    sample_times: list[float] = []

    for step in range(config.steps + 1):
        if step % config.sample_stride == 0:
            samples_u.append(u.copy())
            samples_v.append(v.copy())
            sample_times.append(step * config.dt)
        if step < config.steps:
            u, v = step_system(u, v, config)

    return {
        "packet": packet,
        "omega": omega,
        "u": np.asarray(samples_u, dtype=np.float64),
        "v": np.asarray(samples_v, dtype=np.float64),
        "times": np.asarray(sample_times, dtype=np.float64),
    }


def smooth_packet_fields(
    u_samples: np.ndarray,
    v_samples: np.ndarray,
    sigma: float,
    omega_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    real_part = gaussian_filter1d(u_samples, sigma=sigma, axis=1, mode="wrap")
    imag_part = gaussian_filter1d(v_samples / omega_ref, sigma=sigma, axis=1, mode="wrap")
    rho = real_part**2 + imag_part**2
    rho = np.clip(rho, 1e-12, None)
    theta = np.unwrap(np.angle(real_part + 1j * imag_part), axis=1)
    theta = np.unwrap(theta, axis=0)
    return rho, theta, np.sqrt(rho)


def build_window_observables(
    run: dict[str, object],
    sigma: float,
    config: SimulationConfig,
) -> dict[str, np.ndarray]:
    rho, theta, amplitude = smooth_packet_fields(run["u"], run["v"], sigma, run["omega"])
    dt_sample = config.dt * config.sample_stride

    rho_t = np.gradient(rho, dt_sample, axis=0)
    theta_t = np.gradient(theta, dt_sample, axis=0)
    theta_x = np.gradient(theta, config.dx, axis=1)
    flux_base = rho * theta_x
    flux_div = np.gradient(flux_base, config.dx, axis=1)
    amplitude_xx = np.gradient(np.gradient(amplitude, config.dx, axis=1), config.dx, axis=1)

    threshold = config.mask_fraction * np.max(rho, axis=1, keepdims=True)
    mask = rho > np.maximum(threshold, 1e-9)
    core_slice = (slice(4, -4), slice(4, -4))
    return {
        "rho": rho[core_slice],
        "theta": theta[core_slice],
        "amplitude": amplitude[core_slice],
        "rho_t": rho_t[core_slice],
        "theta_t": theta_t[core_slice],
        "theta_x": theta_x[core_slice],
        "flux_div": flux_div[core_slice],
        "amplitude_xx": amplitude_xx[core_slice],
        "mask": mask[core_slice],
        "times": run["times"][1:-1],
    }


def fit_transport_coefficient(observables: list[dict[str, np.ndarray]]) -> tuple[float, list[float]]:
    x_terms: list[np.ndarray] = []
    y_terms: list[np.ndarray] = []
    per_packet: list[float] = []

    for obs in observables:
        x = obs["flux_div"][obs["mask"]]
        y = -obs["rho_t"][obs["mask"]]
        denom = float(np.dot(x, x))
        lam = float(np.dot(x, y) / max(denom, 1e-12))
        per_packet.append(lam)
        x_terms.append(x)
        y_terms.append(y)

    x_all = np.concatenate(x_terms)
    y_all = np.concatenate(y_terms)
    lam_global = float(np.dot(x_all, y_all) / max(np.dot(x_all, x_all), 1e-12))
    return lam_global, per_packet


def evaluate_window(
    runs: list[dict[str, object]],
    sigma: float,
    config: SimulationConfig,
) -> tuple[dict[str, object], list[dict[str, np.ndarray]]]:
    observables = [build_window_observables(run, sigma, config) for run in runs]
    lam_global, lam_per_packet = fit_transport_coefficient(observables)
    hbar_eff_gauge_m1 = lam_global
    nu_eff = 0.5 * lam_global

    continuity_num_sq = 0.0
    continuity_den_sq = 0.0
    hj_num_sq = 0.0
    hj_den_sq = 0.0
    packet_metrics: list[dict[str, float]] = []

    for run, obs, lam_packet in zip(runs, observables, lam_per_packet, strict=True):
        mask = obs["mask"]
        continuity_residual = obs["rho_t"] + lam_global * obs["flux_div"]
        continuity_scale = np.sqrt(np.mean(obs["rho_t"][mask] ** 2)) + np.sqrt(np.mean((lam_global * obs["flux_div"][mask]) ** 2))
        continuity_rms = float(np.sqrt(np.mean(continuity_residual[mask] ** 2)))
        continuity_norm = float(continuity_rms / max(continuity_scale, 1e-12))

        quantum_structure = obs["theta_x"] ** 2 - obs["amplitude_xx"] / np.clip(obs["amplitude"], 1e-8, None)
        hj_base = hbar_eff_gauge_m1 * obs["theta_t"] + 0.5 * hbar_eff_gauge_m1**2 * quantum_structure
        potential_offset = -float(np.mean(hj_base[mask]))
        hj_residual = hj_base + potential_offset
        hj_scale = np.sqrt(np.mean((hbar_eff_gauge_m1 * obs["theta_t"][mask]) ** 2)) + np.sqrt(
            np.mean((0.5 * hbar_eff_gauge_m1**2 * quantum_structure[mask]) ** 2)
        )
        hj_rms = float(np.sqrt(np.mean(hj_residual[mask] ** 2)))
        hj_norm = float(hj_rms / max(hj_scale, 1e-12))

        continuity_num_sq += float(np.mean(continuity_residual[mask] ** 2))
        continuity_den_sq += continuity_scale**2
        hj_num_sq += float(np.mean(hj_residual[mask] ** 2))
        hj_den_sq += hj_scale**2

        packet_metrics.append(
            {
                "label": run["packet"].label,
                "lambda_packet": float(lam_packet),
                "continuity_residual": continuity_norm,
                "hj_residual": hj_norm,
                "potential_offset": potential_offset,
            }
        )

    continuity_global = float(np.sqrt(continuity_num_sq / max(continuity_den_sq, 1e-12)))
    hj_global = float(np.sqrt(hj_num_sq / max(hj_den_sq, 1e-12)))
    combined = float(np.sqrt(0.5 * (continuity_global**2 + hj_global**2)))
    lambda_std = float(np.std(lam_per_packet))
    lambda_rel_std = float(lambda_std / max(abs(lam_global), 1e-12))
    packet_independent = bool(lambda_rel_std < 0.15)

    return {
        "sigma": sigma,
        "lambda_global": lam_global,
        "nu_eff": nu_eff,
        "hbar_eff_gauge_m1": hbar_eff_gauge_m1,
        "lambda_std": lambda_std,
        "lambda_relative_std": lambda_rel_std,
        "continuity_residual": continuity_global,
        "hj_residual": hj_global,
        "combined_residual": combined,
        "packet_independent": packet_independent,
        "packet_metrics": packet_metrics,
    }, observables


def choose_best_window(window_metrics: list[dict[str, object]]) -> dict[str, object]:
    return min(window_metrics, key=lambda item: (item["combined_residual"], item["lambda_relative_std"]))


def make_residual_plot(window_metrics: list[dict[str, object]], output_dir: Path) -> None:
    sigmas = [item["sigma"] for item in window_metrics]
    continuity = [item["continuity_residual"] for item in window_metrics]
    hamilton_jacobi = [item["hj_residual"] for item in window_metrics]
    combined = [item["combined_residual"] for item in window_metrics]

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    ax.plot(sigmas, continuity, marker="o", lw=2.2, color="#1d3557", label="Continuity residual")
    ax.plot(sigmas, hamilton_jacobi, marker="s", lw=2.2, color="#c1121f", label="Hamilton-Jacobi residual")
    ax.plot(sigmas, combined, marker="^", lw=2.4, color="#2a9d8f", label="Combined residual")
    ax.set_xlabel("coarse-graining sigma W")
    ax.set_ylabel("normalized RMS residual")
    ax.set_title("Madelung residuals vs coarse-graining scale")
    ax.grid(alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "residuals_vs_window.png", dpi=180)
    plt.close(fig)


def make_transport_plot(window_metrics: list[dict[str, object]], output_dir: Path) -> None:
    sigmas = [item["sigma"] for item in window_metrics]
    global_lambda = [item["lambda_global"] for item in window_metrics]

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    ax.plot(sigmas, global_lambda, marker="o", lw=2.4, color="#3a86ff", label="global lambda = hbar/m")
    for packet_idx, label in enumerate(item["label"] for item in window_metrics[0]["packet_metrics"]):
        packet_values = [wm["packet_metrics"][packet_idx]["lambda_packet"] for wm in window_metrics]
        ax.plot(sigmas, packet_values, lw=1.6, alpha=0.75, label=f"{label} packet")
    ax.set_xlabel("coarse-graining sigma W")
    ax.set_ylabel("transport coefficient lambda")
    ax.set_title("Packet robustness of the emergent transport coefficient")
    ax.grid(alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "transport_coefficient_vs_window.png", dpi=180)
    plt.close(fig)


def make_snapshot_plot(
    best_window: dict[str, object],
    best_observables: list[dict[str, np.ndarray]],
    runs: list[dict[str, object]],
    config: SimulationConfig,
    output_dir: Path,
) -> None:
    packet_index = 0
    obs = best_observables[packet_index]
    run = runs[packet_index]
    frame_index = len(obs["times"]) // 2
    x = np.arange(4, config.nx - 4, dtype=np.float64) * config.dx
    mask = obs["mask"][frame_index]
    continuity_rhs = -best_window["lambda_global"] * obs["flux_div"][frame_index]
    continuity_lhs = obs["rho_t"][frame_index]

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 10.2), constrained_layout=True)
    axes[0].plot(x, run["u"][frame_index + 4][4:-4], color="#6d597a", lw=1.5, label="raw field u")
    axes[0].plot(x, obs["rho"][frame_index], color="#2a9d8f", lw=2.0, label="coarse rho")
    axes[0].set_title(f"Representative packet snapshot at W={best_window['sigma']:.1f} ({run['packet'].label})")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, obs["theta"][frame_index], color="#1d3557", lw=2.0, label="phase theta")
    axes[1].fill_between(x, obs["theta"][frame_index], where=mask, alpha=0.18, color="#3a86ff", label="active mask")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(x, continuity_lhs, color="#c1121f", lw=2.0, label="rho_t")
    axes[2].plot(x, continuity_rhs, color="#264653", lw=2.0, ls="--", label="-lambda d_x(rho theta_x)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    axes[2].set_xlabel("x")
    fig.savefig(output_dir / "best_window_snapshot.png", dpi=180)
    plt.close(fig)


def write_metrics(metrics: dict[str, object], out_dir: Path) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_report(metrics: dict[str, object], out_dir: Path) -> None:
    best = metrics["best_window"]
    packet_lines = [
        f"- {packet['label']}: lambda={packet['lambda_packet']:.4f}, continuity={packet['continuity_residual']:.4f}, HJ={packet['hj_residual']:.4f}"
        for packet in best["packet_metrics"]
    ]
    sigma_table = [
        "| W | lambda | nu_eff | continuity | HJ | combined | rel std | packet independent |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in metrics["window_scan"]:
        sigma_table.append(
            f"| {item['sigma']:.1f} | {item['lambda_global']:.4f} | {item['nu_eff']:.4f} | {item['continuity_residual']:.4f} | {item['hj_residual']:.4f} | {item['combined_residual']:.4f} | {item['lambda_relative_std']:.4f} | {item['packet_independent']} |"
        )

    positive_result = (
        best["combined_residual"] < metrics["report_residual_threshold"]
        and best["packet_independent"]
    )

    interpretation = (
        "The coarse-grained Madelung closure is quantitatively credible in this window: the transport coefficient is stable across packet families and the combined residual is below the reporting threshold."
        if positive_result
        else "This run does not establish a full Madelung emergence result. The continuity equation may still look reasonable, but the Hamilton-Jacobi closure remains too loose and the result should be treated as partial or negative evidence."
    )

    report = rf"""# Report: Sine-Gordon to Madelung Coarse-Graining

## Setup
- Substrate: deterministic 1D sine-Gordon chain with damping-free-to-weakly-damped self-mixing
- Grid: {metrics['nx']} sites, dx={metrics['dx']:.2f}
- Time step: dt={metrics['dt']:.3f}
- Steps: {metrics['steps']}
- Sample stride: every {metrics['sample_stride']} steps
- Substrate parameters: c={metrics['c']:.2f}, alpha={metrics['alpha']:.2f}, gamma={metrics['gamma']:.4f}, K={metrics['background_k']:.2f}
- Packet family count: {len(metrics['packet_suite'])}

## Method
- Simulate three deterministic packet families on the same substrate.
- Build a coarse complex field from the filtered pair $u + i v / \omega_0$ with a fixed reference frequency.
- Extract $\rho$, $\theta$, and $R=\sqrt{{\rho}}$ for each coarse-graining scale $W$.
- Fit the identifiable transport invariant $\lambda = \hbar_{{eff}} / m_{{eff}}$ from the continuity equation using a single global value across all packets.
- Use the unit convention $m_{{eff}}=1$ only for reporting the gauge-fixed quantity $\hbar_{{eff}}=\lambda$; the physically identifiable invariant is $\nu_{{eff}} = \hbar_{{eff}} / 2 m_{{eff}} = \lambda / 2$.

## Window Scan
{chr(10).join(sigma_table)}

## Best Window
- Best coarse-graining scale: W={best['sigma']:.1f}
- Global lambda = hbar_eff / m_eff: {best['lambda_global']:.4f}
- Derived nu_eff = hbar_eff / 2m_eff: {best['nu_eff']:.4f}
- Gauge-fixed hbar_eff at m_eff=1: {best['hbar_eff_gauge_m1']:.4f}
- Continuity residual: {best['continuity_residual']:.4f}
- Quantum Hamilton-Jacobi residual: {best['hj_residual']:.4f}
- Combined residual: {best['combined_residual']:.4f}
- Packet independence check (relative std of lambda): {best['lambda_relative_std']:.4f}

## Packet-Level Robustness At Best Window
{chr(10).join(packet_lines)}

## Interpretation
{interpretation}

The central honest point is identifiability. This experiment directly constrains the transport coefficient $\lambda = \hbar_{{eff}} / m_{{eff}}$ from the coarse dynamics. It does **not** independently identify absolute $\hbar_{{eff}}$ and $m_{{eff}}$ without an external calibration or an additional dispersion-relation measurement. That is a meaningful partial result: the substrate can still generate the Madelung transport scale even if the absolute quantum units remain gauge-dependent in this first pass.

## Verdict
This run {'supports a partial Madelung emergence result' if positive_result else 'does not yet establish a full Madelung emergence result'} on this parameter set. The decisive quantity is the best combined residual {best['combined_residual']:.4f} at W={best['sigma']:.1f}, together with lambda robustness {best['lambda_relative_std']:.4f} across the packet family.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def run(config: SimulationConfig, out_dir: Path) -> dict[str, object]:
    output_dir = out_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    packets = default_packets()
    runs = [run_packet(config, packet) for packet in packets]

    window_scan: list[dict[str, object]] = []
    best_observables: list[dict[str, np.ndarray]] | None = None
    best_metrics: dict[str, object] | None = None

    for sigma in config.coarse_sigmas:
        metrics, observables = evaluate_window(runs, sigma, config)
        window_scan.append(metrics)
        if best_metrics is None or (metrics["combined_residual"], metrics["lambda_relative_std"]) < (
            best_metrics["combined_residual"],
            best_metrics["lambda_relative_std"],
        ):
            best_metrics = metrics
            best_observables = observables

    assert best_metrics is not None
    assert best_observables is not None

    make_residual_plot(window_scan, output_dir)
    make_transport_plot(window_scan, output_dir)
    make_snapshot_plot(best_metrics, best_observables, runs, config, output_dir)

    metrics: dict[str, object] = {
        "nx": config.nx,
        "dx": config.dx,
        "dt": config.dt,
        "steps": config.steps,
        "sample_stride": config.sample_stride,
        "c": config.c,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "background_k": config.background_k,
        "omega_ref": config.omega_ref,
        "packet_suite": [packet.__dict__ for packet in packets],
        "report_residual_threshold": config.report_residual_threshold,
        "window_scan": window_scan,
        "best_window": best_metrics,
    }
    write_metrics(metrics, out_dir)
    write_report(metrics, out_dir)
    return metrics


def parse_args() -> tuple[SimulationConfig, Path]:
    parser = argparse.ArgumentParser(description="Run the sine-Gordon to Madelung coarse-graining test.")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    return SimulationConfig(steps=args.steps), args.out_dir


def main() -> None:
    config, out_dir = parse_args()
    metrics = run(config, out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()