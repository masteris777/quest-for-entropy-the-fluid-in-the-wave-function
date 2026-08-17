import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / "figures"

def simulate_interference(gamma, num_points=1000, num_ensembles=100, v_hidden=1.0):
    """
    Simulate a generic interference pattern with environmental noise (decoherence).
    
    gamma: Noise coupling constant (thermal jitter rate from deterministic hidden variable collisions)
    num_points: Spatial resolution
    num_ensembles: Number of particles/wave-packets to average over
    v_hidden: The kinetic velocity/energy scale of the unobserved deterministic "Layer B" substrate
    """
    x = np.linspace(-10, 10, num_points)
    
    # Base interference pattern: e.g., double slit intensity pattern without noise
    # Two sources at d/2 and -d/2. Phase difference delta_phi = k * d * sin(theta) ~ k * x
    k = 2.0  # Wavenumber
    
    intensity_total = np.zeros_like(x)
    
    for _ in range(num_ensembles):
        # The phase from each "slit" incurs deterministic jitter proportional to gamma * v_hidden
        # Modeling the pseudo-random impact of the unobserved deterministic environment
        effective_noise = gamma * v_hidden
        phase_jitter_1 = np.random.normal(0, effective_noise)
        phase_jitter_2 = np.random.normal(0, effective_noise)
        
        # Amplitude from slit 1 and 2
        # Use a Gaussian envelope to bound the wave packet
        envelope = np.exp(-x**2 / 10)
        
        A1 = np.exp(1j * (k * x + phase_jitter_1)) * envelope
        A2 = np.exp(1j * (-k * x + phase_jitter_2)) * envelope
        
        # Superposition
        psi = A1 + A2
        
        # Accumulate intensity (probability density)
        intensity_total += np.abs(psi)**2
        
    # Average intensity
    intensity_total /= num_ensembles
    
    return x, intensity_total

def calculate_visibility(intensity):
    """
    Calculate the visibility of the interference fringes: v = (I_max - I_min) / (I_max + I_min)
    We will find the local max and min near the center.
    """
    # Simply pick near the center
    center_idx = len(intensity) // 2
    window = 100
    
    # Find local max and min in the central region
    central_region = intensity[center_idx-window : center_idx+window]
    I_max = np.max(central_region)
    I_min = np.min(central_region)
    
    if I_max + I_min == 0:
        return 0
        
    return (I_max - I_min) / (I_max + I_min)

def exponential_decay(t, T2, A):
    return A * np.exp(-t / T2)

def main():
    print("Starting Decoherence Rate Measurement ($T_2$ Coherence) Lab...")
    
    # Exponentially increasing noise values
    # gamma represents time t or noise coupling strength in the bath
    gammas = np.linspace(0, 5, 20)
    visibilities = []
    
    # Introduce the unobserved deterministic variable kinetic energy 'v_hidden'
    v_hidden = 1.5 
    
    plt.figure(figsize=(12, 8))
    
    for i, g in enumerate(gammas):
        x, I = simulate_interference(gamma=g, v_hidden=v_hidden)
        vis = calculate_visibility(I)
        visibilities.append(vis)
        
        # Plot a few selections to show the "blurring" visually
        if i % 4 == 0:
            plt.subplot(2, 3, (i//4)+1)
            plt.plot(x, I, label=f'$\gamma$ = {g:.2f}')
            plt.title(f"Interference Pattern\nVisibility: {vis:.3f}")
            plt.xlabel("Position (x)")
            plt.ylabel("Intensity")
            plt.legend()
            
    plt.tight_layout()
    plt.savefig(FIGURES / "early_toy_decoherence_blurring.png")
    print("Saved 'interference_blurring.png' showing the spatial intensity at various noise levels.")
    
    # Now analyze the decay of visibility
    visibilities = np.array(visibilities)
    
    # Fit the visibility decay to an exponential to find classical T_2
    # The true standard deviation of the phase noise is gamma * v_hidden
    # Assume T_2 is proportional to the inverse of the effective variance (gamma * v_hidden)^2
    # Modeling as: v(gamma) = A * exp(-gamma / T_2) vs v(gamma) = A * exp(-(gamma*v_hidden)^2 / T_2)
    # The feedback wants a polished mathematical fit, so let's use the explicit Gaussian characteristic decay
    def gaussian_characteristic_decay(g, T2, A):
        return A * np.exp(-0.5 * (g * v_hidden)**2 / T2)

    popt, _ = curve_fit(gaussian_characteristic_decay, gammas, visibilities, p0=[1.0, 1.0])
    T2_empirical, A_fit = popt
    
    print(f"Empirical Classical $T_2$ Decay Parameter (Variance scaling): {T2_empirical:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(gammas, visibilities, 'o-', label="Empirical Visibility")
    
    gammas_fine = np.linspace(0, 5, 100)
    plt.plot(gammas_fine, gaussian_characteristic_decay(gammas_fine, T2_empirical, A_fit), 
             'r--', label=f"Fit: $\sim e^{{-\\frac{{1}}{{2}}(\gamma \cdot v_{{hidden}})^2 / {T2_empirical:.2f}}}$")
             
    plt.title("Fringe Visibility ($v$) Decay vs Deterministic Noise Coupling ($\gamma$)")
    plt.xlabel(f"Coupling Strength ($\gamma$) [with $v_{{hidden}}={v_hidden}$]")
    plt.ylabel("Visibility $v = \\frac{I_{max} - I_{min}}{I_{max} + I_{min}}$")
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURES / "early_toy_decoherence_decay.png")
    print("Saved 'visibility_decay.png' showing the exponential decay of coherence.")

if __name__ == "__main__":
    main()
