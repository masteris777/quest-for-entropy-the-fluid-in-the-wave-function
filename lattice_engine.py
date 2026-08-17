import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def generate_bath(nx, dx, c, K, alpha, num_modes, T_eff):
    # Generates a deterministic high-frequency bath
    x = np.arange(nx) * dx
    u_bath = np.zeros(nx)
    v_bath = np.zeros(nx)
    
    np.random.seed(42) # fixed deterministic seed
    
    # We want high frequency modes (to not overlap with kink's low-k profile)
    # Kink width is ~ c/sqrt(alpha)
    k_min = 2.0 * np.sqrt(alpha) / c 
    k_max = np.pi / dx # Nyquist
    
    ks = np.random.uniform(k_min, k_max, num_modes)
    phases = np.random.uniform(0, 2*np.pi, num_modes)
    
    # Total energy = sum 0.5 * (v^2 + omega^2 u^2) ~ T_eff
    # Since we sum many random modes, the amplitude of each should scale carefully
    A_k = np.sqrt(2 * T_eff / num_modes) / np.sqrt(np.mean(c**2 * ks**2 + K + alpha))
    
    for k, phi in zip(ks, phases):
        omega_k = np.sqrt(c**2 * k**2 + K + alpha)
        # Add to background
        u_bath += A_k * np.cos(k * x + phi)
        v_bath -= A_k * omega_k * np.sin(k * x + phi)
        
    # ensure boundary conditions are strictly 0 for the bath so we don't mess up the fixed BCs
    # Apply a window function that tapers to 0 at boundaries
    window = np.sin(np.pi * x / (nx * dx))**2
    u_bath *= window
    v_bath *= window
    
    return u_bath, v_bath

def run_sg_1d_kink(
    c=1.0, K=0.0, alpha=0.1, gamma=0.005,
    nx=2048, dx=0.5, dt=0.2, steps=20000,
    center=None, T_eff=0.0, num_modes=100,
    sample_stride=20
):
    if center is None:
        center = (nx * dx) / 2.0
        
    x = np.arange(nx) * dx
    
    # 1. Initialize Kink
    # 4 arctan(exp(sqrt(alpha/c^2) (x - x0)))
    gamma_kink = np.sqrt(alpha / c**2)
    u_kink = 4.0 * np.arctan(np.exp(gamma_kink * (x - center)))
    v_kink = np.zeros(nx)
    
    # 2. Add Bath
    u_bath, v_bath = generate_bath(nx, dx, c, K, alpha, num_modes, T_eff)
    
    u = u_kink + u_bath
    v = v_kink + v_bath
    
    # Fixed boundary conditions
    u[0] = 0.0
    u[-1] = 2.0 * np.pi
    v[0] = 0.0
    v[-1] = 0.0
    
    num_samples = steps // sample_stride
    times = np.zeros(num_samples)
    X_pos = np.zeros(num_samples)
    
    sample_idx = 0
    k2 = (c / dx)**2
    
    for step in range(steps):
        if step % sample_stride == 0 and sample_idx < num_samples:
            times[sample_idx] = step * dt
            # Find kink position X(t) by locating where u crosses pi
            crossings = np.where(u > np.pi)[0]
            if len(crossings) > 0 and crossings[0] > 0:
                idx = crossings[0]
                x0 = x[idx-1]
                x1 = x[idx]
                u0_val = u[idx-1]
                u1_val = u[idx]
                # Linear interpolation
                X_pos[sample_idx] = x0 + (np.pi - u0_val) * (x1 - x0) / (u1_val - u0_val)
            else:
                X_pos[sample_idx] = center # Fallback
                
            sample_idx += 1
            
        lap_u = np.zeros(nx)
        lap_u[1:-1] = u[2:] - 2.0*u[1:-1] + u[:-2]
        force_0 = k2 * lap_u - alpha * np.sin(u) - gamma * v
        force_0[0] = 0; force_0[-1] = 0
        
        v_half = v + 0.5 * dt * force_0
        u = u + dt * v_half
        u[0] = 0; u[-1] = 2.0*np.pi
        
        lap_u1 = np.zeros(nx)
        lap_u1[1:-1] = u[2:] - 2.0*u[1:-1] + u[:-2]
        force_1 = k2 * lap_u1 - alpha * np.sin(u) - gamma * v_half
        force_1[0] = 0; force_1[-1] = 0
        
        v = v_half + 0.5 * dt * force_1
        v[0] = 0; v[-1] = 0
        
    return times, X_pos

def calculate_msd(X_pos):
    N = len(X_pos)
    max_lag = N // 2
    msd = np.zeros(max_lag)
    for tau in range(1, max_lag):
        # time average over the single trajectory (ergodicity)
        diff = X_pos[tau:] - X_pos[:-tau]
        msd[tau] = np.mean(diff**2)
    return msd

def main():
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    T_eff_list = [0.01, 0.03, 0.05]
    results = {}
    
    plt.figure()
    for T in T_eff_list:
        print(f"Running T_eff = {T}")
        times, X_pos = run_sg_1d_kink(T_eff=T, steps=40000, sample_stride=20, nx=2048)
        
        # Plot trajectory offset to 0 for comparing
        plt.plot(times, X_pos - X_pos[0], label=f"T_eff={T}")
        
        msd = calculate_msd(X_pos)
        
        # Fit diffusion constant in the later lag times (diffusion regime)
        # Lag times tau * dt * sample_stride
        lags = np.arange(len(msd)) * (0.2 * 20)
        
        fit_idx = len(msd) // 4
        # Fit MSD = 2 D t -> D = MSD / (2t)
        D = np.mean(msd[fit_idx:] / (2 * lags[fit_idx:]))
        results[str(T)] = float(D)
        
        # Save MSD array for later plotting
        np.save(out_dir / f"msd_T_{T}.npy", msd)
        np.save(out_dir / f"X_pos_T_{T}.npy", X_pos)
        
    plt.xlabel("Time")
    plt.ylabel("Displacement Delta X(t)")
    plt.title("Deterministic Brownian Random Walk of Kink")
    plt.legend()
    plt.savefig(out_dir / "kink_trajectory.png")
    plt.close()
    
    # Plot MSD
    plt.figure()
    for T in T_eff_list:
        msd = np.load(out_dir / f"msd_T_{T}.npy")
        lags = np.arange(len(msd)) * (0.2 * 20)
        plt.plot(lags[1:], msd[1:], label=f"T_eff={T}")
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Lag Time tau")
    plt.ylabel("MSD <Delta X(tau)^2>")
    plt.title("Mean Squared Displacement vs Lag Time")
    plt.legend()
    plt.savefig(out_dir / "msd_vs_tau.png")
    plt.close()
    
    with open(out_dir.parent / "metrics_13.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
