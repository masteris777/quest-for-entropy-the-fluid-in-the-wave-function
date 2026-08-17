import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros

def generate_mode(n, m, grid_size=300):
    R = 1.0
    x = np.linspace(-R, R, grid_size)
    y = np.linspace(-R, R, grid_size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    k_vals = jn_zeros(m, n)
    k = k_vals[-1]
    
    psi = jn(m, k * r) * np.cos(m * theta)
    psi[r > R] = 0
    density = psi**2
    return X, Y, density

def plot_harmonics():
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Acoustic Fluid Standing Waves (Quantized Orbitals)', fontsize=16)
    
    modes = [
        (1, 0, 's-like orbital (n=1, m=0)'),
        (1, 1, 'p-like orbital (n=1, m=1)'),
        (2, 0, '2s-like (radial node) (n=2, m=0)'),
        (1, 2, 'd-like (angular nodes) (n=1, m=2)')
    ]
    
    for ax, (n, m, title) in zip(axes.flatten(), modes):
        X, Y, density = generate_mode(n, m)
        c = ax.pcolormesh(X, Y, density, shading='auto', cmap='inferno')
        circle = plt.Circle((0, 0), 1.0, color='w', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(title, pad=15)
        ax.axis('off')
    
    plt.tight_layout()
    output_path = str(__import__("pathlib").Path(__file__).resolve().parent
                       / "figures" / "early_toy_orbitals.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved standing wave harmonics to {output_path}")

if __name__ == '__main__':
    print("Generating discrete acoustic fluid standing waves...")
    plot_harmonics()
    print("Done!")
