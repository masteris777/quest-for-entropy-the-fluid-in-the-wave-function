"""The zipper: one complex wave function = a density and a flow.

    uv run python the_zip.py

A wave packet sloshing in a harmonic trap, solved exactly the standard way
(split-step FFT on the Schrodinger equation, hbar = m = omega = 1):

    i d(psi)/dt = -1/2 d2(psi)/dx2 + 1/2 x^2 psi

The packet starts displaced AND too wide for the trap, so it does two things at
once: it slides from side to side, and it breathes in and out. That is on
purpose - a packet that only slides has a density that never changes shape and a
flow that is the same everywhere, so you cannot see the two halves of the zip
doing different jobs. This one you can.

Top    - the wave function as it is usually written: TWO real curves, the real
         and imaginary parts, doing an endless quarter-turn dance. The dashed
         outline around them is |psi|, the envelope they are trapped inside.
Middle - the SAME object read as a fluid: density rho = |psi|^2 (the dashed
         outline above, squared), with arrows for the flow riding on top of it.
Bottom - the flow itself, v = j / rho (j = the probability current), as a curve.

Nothing is fitted or illustrated; all three panels are the same solved wave
function, read three ways.

WHAT THE ANIMATION MUST SHOW (visualization, no numbers quoted from it)
  V1  Two real curves churning into each other inside a slow-moving envelope.
  V2  That envelope, squared, IS the lump of fluid in the middle panel.
  V3  Density change and flow change are visibly different things: the arrows
      point outward while the lump is fattening, inward while it is thinning,
      and lean the way the whole lump is travelling.
"""

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
ASSETS = HERE / "figures"

N = 1024                 # grid points
L = 16.0                 # box half-width
X0 = 2.0                 # initial displacement
SQUEEZE = 1.45           # initial width, in units of the trap's natural width
DT = 0.02
N_STEP = 320             # one full slosh period is 2*pi = 314 steps
EVERY = 4
N_ARROW = 5

INK = "#0a1024"
GRID = "#2a3a5f"
LABEL = "#9fb4d8"
RE_C = "#7fc6cf"
IM_C = "#ff9d5c"
ENV_C = "#dfe9ff"
RHO_C = "#2a6a8f"
FLOW_C = "#eafcff"


def simulate():
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

    s = SQUEEZE
    psi = (1.0 / (np.pi * s ** 2)) ** 0.25 * np.exp(-((x - X0) ** 2) / (2.0 * s ** 2))
    V = 0.5 * x ** 2

    half_V = np.exp(-0.5j * V * DT)
    full_K = np.exp(-0.5j * (k ** 2) * DT)

    frames = []
    for n in range(N_STEP):
        psi = half_V * psi
        psi = np.fft.ifft(full_K * np.fft.fft(psi))
        psi = half_V * psi
        if n % EVERY == 0:
            frames.append(psi.copy())
    return x, dx, frames


def density_and_flow(psi, dx):
    rho = np.abs(psi) ** 2
    dpsi = np.gradient(psi, dx)
    j = np.imag(np.conj(psi) * dpsi)          # probability current (hbar = m = 1)
    v = np.full_like(rho, np.nan)
    ok = rho > 0.02 * rho.max()                # flow is meaningless where nothing is
    v[ok] = j[ok] / rho[ok]
    return rho, v


def render(x, dx, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    amp = max(float(np.abs(p).max()) for p in frames)
    rho_max = amp ** 2
    vmax = max(float(np.nanmax(np.abs(density_and_flow(p, dx)[1]))) for p in frames)

    fig, (top, mid, bot) = plt.subplots(
        3, 1, figsize=(7.6, 6.0), facecolor=INK,
        gridspec_kw={"hspace": 0.22, "height_ratios": [1.0, 1.0, 0.8]})
    for a in (top, mid, bot):
        a.set_facecolor(INK)
        a.set_xlim(-6.5, 6.5)
        a.set_xticks([])
        a.set_yticks([])
        a.axhline(0.0, color=GRID, lw=0.9)
        a.tick_params(colors=LABEL)
        for s in a.spines.values():
            s.set_color(GRID)

    top.set_ylim(-1.15 * amp, 1.15 * amp)
    (re_line,) = top.plot([], [], color=RE_C, lw=2.0)
    (im_line,) = top.plot([], [], color=IM_C, lw=2.0)
    (env_up,) = top.plot([], [], color=ENV_C, lw=1.1, ls="--", alpha=0.75)
    (env_dn,) = top.plot([], [], color=ENV_C, lw=1.1, ls="--", alpha=0.75)
    top.set_ylabel("the two curves\nwe write down", color=LABEL, fontsize=9)

    mid.set_ylim(-0.12 * rho_max, 1.34 * rho_max)
    fill = mid.fill_between(x, 0, np.zeros_like(x), color=RHO_C, alpha=0.85)
    arrows = mid.quiver(np.zeros(N_ARROW), np.full(N_ARROW, 1.16 * rho_max),
                        np.zeros(N_ARROW), np.zeros(N_ARROW),
                        color=FLOW_C, angles="xy", scale_units="xy", scale=1.0,
                        width=0.005, headwidth=4.0)
    mid_centre = mid.axvline(0.0, color=ENV_C, lw=0.9, ls=":", alpha=0.5)
    mid.set_ylabel("how much\nis here", color=LABEL, fontsize=9)

    bot.set_ylim(-1.15 * vmax, 1.15 * vmax)
    flow_lc = LineCollection([], linewidths=2.2)
    bot.add_collection(flow_lc)
    bot_centre = bot.axvline(0.0, color=ENV_C, lw=0.9, ls=":", alpha=0.5)
    bot.set_ylabel("which way\nit flows", color=LABEL, fontsize=9)
    bot.set_xlabel("position", color=LABEL, fontsize=9)
    bot.set_yticks([-0.75 * vmax, 0.0, 0.75 * vmax])
    bot.set_yticklabels(["flowing left", "still", "flowing right"],
                        color=LABEL, fontsize=8)

    fig.subplots_adjust(left=0.19, right=0.985, top=0.97, bottom=0.09, hspace=0.22)

    arrow_scale = 0.62 / vmax        # longest arrow spans ~0.6 units of position
    flow_rgb = mcolors.to_rgb(FLOW_C)

    def draw(i):
        nonlocal fill
        psi = frames[i]
        re_line.set_data(x, psi.real)
        im_line.set_data(x, psi.imag)
        env = np.abs(psi)
        env_up.set_data(x, env)
        env_dn.set_data(x, -env)

        rho, v = density_and_flow(psi, dx)
        fill.remove()
        fill = mid.fill_between(x, 0, rho, color=RHO_C, alpha=0.85)

        # the flow curve fades out with the density: where there is no fluid there
        # is no "which way", so the line must not end in a hard edge that reads
        # as if the LENGTH of the line meant something. Only its height does.
        ok = np.isfinite(v)
        pts = np.column_stack([x[ok], v[ok]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        weight = (rho[ok][:-1] / rho.max()) ** 0.35
        cols = np.zeros((len(segs), 4))
        cols[:, :3] = flow_rgb
        cols[:, 3] = np.clip(weight, 0.0, 1.0)
        flow_lc.set_segments(segs)
        flow_lc.set_color(cols)

        # a row of flow arrows above the lump, sampled only where there IS a lump
        live = np.flatnonzero(rho > 0.06 * rho.max())
        idx = live[np.linspace(0, len(live) - 1, N_ARROW).astype(int)]
        arrows.set_offsets(np.column_stack([x[idx], np.full(N_ARROW, 1.16 * rho_max)]))
        arrows.set_UVC(v[idx] * arrow_scale, np.zeros(N_ARROW))

        centre = float(np.sum(rho * x) * dx)
        mid_centre.set_xdata([centre, centre])
        bot_centre.set_xdata([centre, centre])
        return [re_line, im_line, env_up, env_dn, fill, arrows, flow_lc,
                mid_centre, bot_centre]

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "the_zip.gif"
    anim.save(out, writer=PillowWriter(fps=14))
    draw(int(len(frames) * 0.18))
    fig.savefig(ASSETS / "the_zip_preview.png", dpi=110, facecolor=INK)
    print(f"peak |psi| {amp:.3f}, peak flow {vmax:.3f}")
    print(f"wrote {out} and the_zip_preview.png")


if __name__ == "__main__":
    x, dx, frames = simulate()
    norm = [float(np.sum(np.abs(p) ** 2) * dx) for p in (frames[0], frames[-1])]
    print(f"norm at start/end: {norm[0]:.6f} / {norm[1]:.6f}  (must stay 1)")
    width = [float(np.sqrt(np.sum(np.abs(p) ** 2 * x ** 2) * dx
                           - (np.sum(np.abs(p) ** 2 * x) * dx) ** 2)) for p in frames]
    print(f"packet width breathes between {min(width):.3f} and {max(width):.3f}")
    render(x, dx, frames)
