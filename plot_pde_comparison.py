import matplotlib.pyplot as plt
import numpy as np

# Data from your benchmarking
simulations = np.arange(1, 11)  # 10 simulations for this demo
python_times = np.array([1125.2] * 10)  # ~11252 seconds for 10 runs, so ~1125.2 per sim
c_times = np.array([1.043] * 10)  # 104.3 seconds for 100 runs, so ~1.043 per sim

# Plot
plt.figure(figsize=(10, 6))

plt.plot(simulations, python_times, label="Python (Numpy)", marker="o", linestyle="--")
plt.plot(simulations, c_times, label="C (OpenMP)", marker="o", linestyle="--")

# Log scale for dramatic effect
plt.yscale("log")

# Labels and Title
plt.xlabel("Simulation Number")
plt.ylabel("Time per Simulation (seconds)")
plt.title("Python vs C Simulation Time Comparison (Heat Diffusion PDE)")

# Annotate speedup
for i, sim in enumerate(simulations):
    speedup = python_times[i] / c_times[i]
    plt.annotate(
        f"{int(speedup)}x",
        (sim, c_times[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        color="green",
    )

plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Save to file
plt.savefig("pde_python_vs_c.png", dpi=300)
plt.show()
