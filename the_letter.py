"""What the letter i is worth: the same equation, with and without it.

    uv run python the_letter.py

One packet, one bowl, one starting condition, two evolutions:

    WITH i     i d(psi)/dt = -1/2 d2(psi)/dx2 + V psi     (Schrodinger)
    WITHOUT i    d(psi)/dt = +1/2 d2(psi)/dx2 - V psi     (the heat equation)

Deleting the i is the only difference between the two runs. Everything else -
grid, trap, initial packet, solver, step size - is identical, and both are run
with the same split-step FFT scheme.

The picture is a space-time diagram: position runs up the page, time runs left
to right, brightness is how much stuff is at that place at that moment.

WITH i     the packet sloshes and keeps sloshing. It never settles.
WITHOUT i  the packet slides to the bottom of the bowl, settles, and stops.
           It also fades away as it goes: the total drops by the factor printed
           at the end of the run. Each column is scaled to its own brightest
           point so the shape stays visible after the fading - otherwise the
           lower panel would simply go black.
"""

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
ASSETS = HERE / "figures"

N = 512
L = 12.0
X0 = 2.2
DT = 0.02
N_STEP = 640             # ~2 full sloshing periods (one period is 2*pi)
KEEP = 2                 # keep every other step as a column of the picture

INK = "#0a1024"
GRID = "#2a3a5f"
LABEL = "#9fb4d8"


def grids():
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    V = 0.5 * x ** 2
    psi0 = (1.0 / np.pi) ** 0.25 * np.exp(-((x - X0) ** 2) / 2.0)
    return x, dx, k, V, psi0


def run(with_i):
    """Same split-step scheme both times; the only change is the factor -1j."""
    x, dx, k, V, psi0 = grids()
    step = -1j if with_i else -1.0
    half_V = np.exp(0.5 * step * V * DT)
    full_K = np.exp(0.5 * step * (k ** 2) * DT)

    psi = psi0.astype(complex)
    cols, mass = [], []
    for n in range(N_STEP):
        psi = half_V * psi
        psi = np.fft.ifft(full_K * np.fft.fft(psi))
        psi = half_V * psi
        if n % KEEP == 0:
            rho = np.abs(psi) ** 2
            mass.append(float(rho.sum() * dx))
            cols.append(rho)
    return x, np.array(cols).T, np.array(mass)      # shape (position, time)


def render(x, sheets, masses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "ink_to_light", [INK, "#1d4a6b", "#3f9bbf", "#eafcff"])

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), facecolor=INK,
                             gridspec_kw={"hspace": 0.30})
    keep = np.abs(x) <= 4.5
    t_end = N_STEP * DT

    for ax, sheet, tag in zip(axes, sheets, ["with the i", "without the i"]):
        col_max = sheet.max(axis=0)
        shown = sheet[keep] / np.where(col_max > 0, col_max, 1.0)
        ax.imshow(shown, origin="lower", aspect="auto", cmap=cmap,
                  extent=[0.0, t_end, x[keep][0], x[keep][-1]],
                  vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax.set_facecolor(INK)
        ax.set_ylabel("position", color=LABEL, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.text(0.15, x[keep][-1] * 0.72, tag, color="#eafcff", fontsize=10)

    axes[1].set_xlabel("time", color=LABEL, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.09)

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "the_letter.png"
    fig.savefig(out, dpi=130, facecolor=INK)
    print(f"wrote {out}")


if __name__ == "__main__":
    x, sheet_i, mass_i = run(with_i=True)
    _, sheet_r, mass_r = run(with_i=False)
    print(f"with i   : total stays {mass_i[0]:.6f} -> {mass_i[-1]:.6f}")
    print(f"without i: total drops {mass_r[0]:.6f} -> {mass_r[-1]:.3e} "
          f"({mass_r[0] / mass_r[-1]:.3g}x smaller)")
    centre_i = (sheet_i * x[:, None]).sum(axis=0) / sheet_i.sum(axis=0)
    centre_r = (sheet_r * x[:, None]).sum(axis=0) / sheet_r.sum(axis=0)
    print(f"with i   : centre still swinging at the end, x = {centre_i[-1]:+.3f} "
          f"(range {centre_i.min():+.3f} to {centre_i.max():+.3f})")
    print(f"without i: centre settled at x = {centre_r[-1]:+.5f}")
    render(x, [sheet_i, sheet_r], [mass_i, mass_r])
