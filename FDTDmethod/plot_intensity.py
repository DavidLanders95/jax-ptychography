import numpy as np
import matplotlib.pyplot as plt

# File parameters
filename = "wavefunction.bin"

# Read binary file
try:
    with open(filename, "rb") as f:
        # Read dimensions
        nx = np.frombuffer(f.read(4), dtype=np.int32)[0]
        ny = np.frombuffer(f.read(4), dtype=np.int32)[0]
        
        # Read phi and psi
        phi = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32).reshape((ny, nx))
        psi = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32).reshape((ny, nx))
        
        print(f"Loaded simulation data: {nx}x{ny}")
        
except FileNotFoundError:
    print(f"Error: {filename} not found. Run the simulation first.")
    exit()

# Compute Intensity |Psi|^2 = phi^2 + psi^2
intensity = phi**2 + psi**2

# Plot
plt.figure(figsize=(12, 6))
plt.imshow(intensity, cmap='inferno', origin='lower', extent=[-2.0, 2.0, -1.125, 1.125])
plt.colorbar(label='Intensity |$\psi$|$^2$')
plt.title("Steady State Relativistic Wave Intensity")
plt.xlabel("x (atomic units)")
plt.ylabel("y (atomic units)")

# Save plot
plt.savefig("steady_state_intensity.png", dpi=150)
print("Saved plot to steady_state_intensity.png")
