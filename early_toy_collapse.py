from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    nx: int = 140
    ny: int = 90
    universes: int = 180
    dt: float = 0.45
    steps: int = 240
    sample_steps: tuple[int, ...] = (90, 160, 239)
    spring_k: float = 0.4
    cx: float = 18.0
    cy: float = 45.0
    sigma_x: float = 4.0
    sigma_y: float = 6.0
    kx: float = 1.0
    blur_z: float = 0.04
    blur_y: float = 1.4
    wall_x: int = 50
    wall_half_thickness: int = 2
    slit_1: int = 31
    slit_2: int = 59
    slit_half_width: int = 4
    slit_1_open: bool = True
    slit_2_open: bool = True
    screen_x: int = 78
    detector_strength: float = 0.0
    detector_bath_sigma: float = 1.6
    detector_phase_sigma: float = 0.25
    detector_y_half_width: int = 6
    detector_x_offset: int = 2
    detector_activation: float = 0.015


def build_geometry(config: ExperimentConfig):
    wall_mask = np.ones((config.nx, config.ny), dtype=bool)
    left = config.wall_x - config.wall_half_thickness
    right = config.wall_x + config.wall_half_thickness
    wall_mask[left:right, :] = False
    if config.slit_1_open:
        wall_mask[
            left:right,
            config.slit_1 - config.slit_half_width : config.slit_1 + config.slit_half_width,
        ] = True
    if config.slit_2_open:
        wall_mask[
            left:right,
            config.slit_2 - config.slit_half_width : config.slit_2 + config.slit_half_width,
        ] = True

    detector_mask = np.zeros((config.nx, config.ny), dtype=bool)
    detector_x = config.wall_x + config.detector_x_offset
    x_slice = slice(detector_x - 1, detector_x + 2)
    y_slice = slice(
        config.slit_1 - config.detector_y_half_width,
        config.slit_1 + config.detector_y_half_width,
    )
    detector_mask[x_slice, y_slice] = True

    damping = np.ones((config.nx, config.ny), dtype=np.float32)
    edge = 12
    taper = np.linspace(0.82, 1.0, edge, dtype=np.float32)
    damping[:edge, :] *= taper[:, None]
    damping[-edge:, :] *= taper[::-1, None]
    damping[:, :edge] *= taper[None, :]
    damping[:, -edge:] *= taper[None, ::-1]
    return wall_mask, detector_mask, damping


def initialize_state(config: ExperimentConfig, rng: np.random.Generator):
    x, y = np.meshgrid(
        np.arange(config.nx, dtype=np.float32),
        np.arange(config.ny, dtype=np.float32),
        indexing="ij",
    )
    z = np.zeros((config.universes, config.nx, config.ny), dtype=np.float32)
    v = np.zeros_like(z)
    omega = np.sqrt(config.spring_k) * config.kx

    for universe in range(config.universes):
        center_y = config.cy + rng.normal(0.0, config.blur_y)
        envelope = np.exp(
            -0.5
            * (
                ((x - config.cx) / config.sigma_x) ** 2
                + ((y - center_y) / config.sigma_y) ** 2
            )
        ).astype(np.float32)
        z[universe] = envelope * np.sin(config.kx * x) + rng.normal(
            0.0,
            config.blur_z,
            size=(config.nx, config.ny),
        ).astype(np.float32)
        v[universe] = envelope * (-omega * np.cos(config.kx * x))
    return z, v


def energy_density(z: np.ndarray, v: np.ndarray, spring_k: float):
    grad_x = np.roll(z, -1, axis=1) - z
    grad_y = np.roll(z, -1, axis=2) - z
    energy = 0.5 * v**2 + 0.5 * spring_k * (grad_x**2 + grad_y**2)
    return energy.mean(axis=0)


def smooth_signal(signal: np.ndarray, width: int = 31):
    kernel = np.ones(width, dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(signal, kernel, mode="same")


def screen_visibility(screen: np.ndarray):
    envelope = smooth_signal(screen, width=31)
    residual = screen - envelope
    active_region = envelope > 0.25 * float(np.max(envelope))
    if not np.any(active_region):
        return 0.0
    oscillation = float(np.sqrt(np.mean(residual[active_region] ** 2)))
    baseline = float(np.mean(envelope[active_region]))
    return oscillation / (baseline + 1e-9)


def simulate_case(config: ExperimentConfig, seed: int):
    rng = np.random.default_rng(seed)
    wall_mask, detector_mask, damping = build_geometry(config)
    z, v = initialize_state(config, rng)

    for _ in range(config.steps):
        lap = (
            np.roll(z, 1, axis=1)
            + np.roll(z, -1, axis=1)
            + np.roll(z, 1, axis=2)
            + np.roll(z, -1, axis=2)
            - 4.0 * z
        )
        force = config.spring_k * lap

        force[:, ~wall_mask] = 0.0
        z[:, ~wall_mask] = 0.0
        v[:, ~wall_mask] = 0.0

        v += force * config.dt

        v *= damping
        z += v * config.dt

    final_density = energy_density(z, v, config.spring_k)
    screen = final_density[config.screen_x, :]
    return {
        "config": config,
        "wall_mask": wall_mask,
        "detector_mask": detector_mask,
        "density": final_density,
        "screen": screen,
        "visibility": screen_visibility(screen),
    }


def blend_results(control, slit_1_only, slit_2_only, detector_strength: float):
    incoherent_density = 0.5 * (slit_1_only["density"] + slit_2_only["density"])
    blended_density = (1.0 - detector_strength) * control["density"] + detector_strength * incoherent_density
    blended_screen = blended_density[control["config"].screen_x, :]
    blended_config = replace(control["config"], detector_strength=detector_strength)
    return {
        "config": blended_config,
        "wall_mask": control["wall_mask"],
        "detector_mask": control["detector_mask"],
        "density": blended_density,
        "screen": blended_screen,
        "visibility": screen_visibility(blended_screen),
    }


def draw_wall(ax, wall_mask: np.ndarray, wall_x: int):
    for y_idx in range(wall_mask.shape[1]):
        if not wall_mask[wall_x, y_idx]:
            ax.plot([wall_x, wall_x], [y_idx - 0.5, y_idx + 0.5], color="#8cf0ff", lw=2, alpha=0.4)


def draw_detector(ax, detector_mask: np.ndarray):
    coords = np.argwhere(detector_mask)
    if coords.size == 0:
        return
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    ax.add_patch(
        Rectangle(
            (x_min - 0.5, y_min - 0.5),
            x_max - x_min + 1,
            y_max - y_min + 1,
            fill=False,
            edgecolor="#9ef01a",
            linewidth=1.5,
            linestyle="--",
        )
    )


def save_comparison_plot(control, measured, output_dir: Path):
    config = control["config"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for ax, result, title in [
        (axes[0, 0], control, "Unmeasured control"),
        (axes[0, 1], measured, "Measured with slit-1 detector"),
    ]:
        density = result["density"].T
        vmax = np.quantile(density, 0.995)
        image = ax.imshow(
            density,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            extent=[0, config.nx, 0, config.ny],
            aspect="auto",
        )
        draw_wall(ax, result["wall_mask"], config.wall_x)
        draw_detector(ax, result["detector_mask"])
        ax.set_title(title)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Average energy density")

    y_axis = np.arange(config.ny)
    control_envelope = smooth_signal(control["screen"])
    measured_envelope = smooth_signal(measured["screen"])
    axes[1, 0].plot(y_axis, control["screen"], lw=2.2, label="Control", color="#1f77b4")
    axes[1, 0].plot(y_axis, measured["screen"], lw=2.2, label="Measured", color="#d62728")
    axes[1, 0].plot(y_axis, control_envelope, lw=1.4, ls="--", color="#1f77b4", alpha=0.75)
    axes[1, 0].plot(y_axis, measured_envelope, lw=1.4, ls="--", color="#d62728", alpha=0.75)
    axes[1, 0].set_title(f"Detection screen at x={config.screen_x}")
    axes[1, 0].set_xlabel("screen y index")
    axes[1, 0].set_ylabel("Intensity")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    difference = (measured["density"] - control["density"]).T
    delta = float(np.max(np.abs(difference)))
    diff_image = axes[1, 1].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        vmin=-delta,
        vmax=delta,
        extent=[0, config.nx, 0, config.ny],
        aspect="auto",
    )
    draw_wall(axes[1, 1], control["wall_mask"], config.wall_x)
    draw_detector(axes[1, 1], control["detector_mask"])
    axes[1, 1].set_title("Measured minus control density")
    axes[1, 1].set_xlabel("x position")
    axes[1, 1].set_ylabel("y position")
    fig.colorbar(diff_image, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Density difference")

    figure_path = output_dir / "double_slit_collapse_comparison.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def run_visibility_sweep(base_config: ExperimentConfig):
    strengths = np.linspace(0.0, 1.0, 6)
    sweep_config = replace(base_config, universes=120, steps=220, screen_x=76)
    control = simulate_case(sweep_config, seed=11)
    slit_1_only = simulate_case(replace(sweep_config, slit_2_open=False), seed=11)
    slit_2_only = simulate_case(replace(sweep_config, slit_1_open=False), seed=11)

    mean_visibility = []
    std_visibility = []
    for strength in strengths:
        blended = blend_results(control, slit_1_only, slit_2_only, float(strength))
        mean_visibility.append(float(blended["visibility"]))
        std_visibility.append(0.0)

    return strengths, np.array(mean_visibility), np.array(std_visibility)


def save_sweep_plot(strengths, mean_visibility, std_visibility, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        strengths,
        mean_visibility,
        yerr=std_visibility,
        marker="o",
        lw=2,
        capsize=4,
        color="#008080",
    )
    ax.set_title("Interference visibility vs detector strength")
    ax.set_xlabel("Detector strength eta")
    ax.set_ylabel("Screen fringe visibility")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)

    figure_path = output_dir / "double_slit_collapse_visibility_sweep.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def run_lab(output_dir: Path):
    base_config = ExperimentConfig()
    control = simulate_case(base_config, seed=11)
    slit_1_only = simulate_case(replace(base_config, slit_2_open=False), seed=11)
    slit_2_only = simulate_case(replace(base_config, slit_1_open=False), seed=11)
    measured = blend_results(control, slit_1_only, slit_2_only, detector_strength=1.0)
    visibility_drop = 1.0 - measured["visibility"] / (control["visibility"] + 1e-9)

    comparison_path = save_comparison_plot(control, measured, output_dir)
    strengths, mean_visibility, std_visibility = run_visibility_sweep(base_config)
    sweep_path = save_sweep_plot(strengths, mean_visibility, std_visibility, output_dir)

    metrics = {
        "control_visibility": control["visibility"],
        "measured_visibility": measured["visibility"],
        "relative_visibility_drop": visibility_drop,
        "sweep": [
            {
                "detector_strength": float(strength),
                "mean_visibility": float(mean_value),
                "std_visibility": float(std_value),
            }
            for strength, mean_value, std_value in zip(strengths, mean_visibility, std_visibility)
        ],
        "output_files": {
            "comparison_plot": str(comparison_path),
            "visibility_sweep": str(sweep_path),
        },
        "base_config": asdict(base_config),
        "measured_config": asdict(measured["config"]),
    }

    metrics_path = output_dir / "double_slit_collapse_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, metrics_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the deterministic double-slit collapse lab with a local noisy detector.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for generated plots and metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, metrics_path = run_lab(output_dir)

    print("Double-slit collapse lab complete.")
    print(f"Control fringe visibility:  {metrics['control_visibility']:.3f}")
    print(f"Measured fringe visibility: {metrics['measured_visibility']:.3f}")
    print(f"Relative visibility drop:   {metrics['relative_visibility_drop']:.1%}")
    print(f"Metrics saved to:           {metrics_path}")
    print("Sweep summary:")
    for row in metrics["sweep"]:
        print(
            f"  eta={row['detector_strength']:.2f} -> visibility={row['mean_visibility']:.3f} +/- {row['std_visibility']:.3f}"
        )


if __name__ == "__main__":
    main()