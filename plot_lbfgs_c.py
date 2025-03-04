import matplotlib.pyplot as plt
import re

# Paste your actual log here as a multiline string
log = """
Iteration 0 = 1210.0000000000, ||grad|| = 1.6466232113e+03
Iteration 1: f = 255.0556331855, ||grad|| = 3.1040941829e+02, step = 9.766e-04
Iteration 2: f = 252.3505568827, ||grad|| = 3.2145176112e+02, step = 1.953e-03
Iteration 3: f = 205.8141494478, ||grad|| = 1.7981410760e+01, step = 1.000e+00
Iteration 4: f = 205.5865230377, ||grad|| = 1.2556562062e+01, step = 1.000e+00
Iteration 5: f = 205.2219091164, ||grad|| = 1.7588190777e+01, step = 1.000e+00
Iteration 6: f = 204.1336335055, ||grad|| = 3.8033840358e+01, step = 1.000e+00
Iteration 7: f = 201.5382119030, ||grad|| = 7.2686348386e+01, step = 1.000e+00
Iteration 8: f = 196.3318712840, ||grad|| = 1.2302079206e+02, step = 1.000e+00
Iteration 9: f = 189.4522684152, ||grad|| = 1.5632674010e+02, step = 1.000e+00
Iteration 10: f = 176.3938378395, ||grad|| = 1.6134773538e+02, step = 1.000e+00
Iteration 11: f = 149.4135520512, ||grad|| = 1.0982147737e+02, step = 1.000e+00
Iteration 12: f = 125.2271128618, ||grad|| = 5.0452401163e+01, step = 1.000e+00
Iteration 13: f = 121.2964714941, ||grad|| = 1.3310931818e+02, step = 5.000e-01
Iteration 14: f = 98.4239158299, ||grad|| = 3.4463239172e+01, step = 1.000e+00
Iteration 15: f = 80.8486111861, ||grad|| = 5.2487716802e+01, step = 1.000e+00
Iteration 16: f = 74.6748974341, ||grad|| = 8.5311637531e+01, step = 1.000e+00
Iteration 17: f = 57.1100029980, ||grad|| = 1.7089833027e+01, step = 1.000e+00
Iteration 18: f = 45.9642325407, ||grad|| = 2.5564462388e+01, step = 1.000e+00
Iteration 19: f = 41.7824920411, ||grad|| = 5.2960531425e+01, step = 5.000e-01
Iteration 20: f = 33.9375031852, ||grad|| = 2.3915340964e+01, step = 1.000e+00
Iteration 21: f = 25.6613636786, ||grad|| = 4.2452701504e+01, step = 1.000e+00
Iteration 22: f = 18.5136255132, ||grad|| = 8.3941146955e+00, step = 1.000e+00
Iteration 23: f = 15.1355581438, ||grad|| = 3.6248545021e+01, step = 5.000e-01
Iteration 24: f = 11.4132568145, ||grad|| = 2.7898825919e+01, step = 1.000e+00
Iteration 25: f = 7.6273070367, ||grad|| = 7.7289767758e+00, step = 1.000e+00
Iteration 26: f = 5.7344740100, ||grad|| = 2.2390735093e+01, step = 2.500e-01
Iteration 27: f = 4.5085829135, ||grad|| = 3.5888354975e+01, step = 1.000e+00
Iteration 28: f = 2.3960109119, ||grad|| = 1.0325369827e+01, step = 1.000e+00
Iteration 29: f = 1.5549688253, ||grad|| = 3.0024586214e+01, step = 1.000e+00
Iteration 30: f = 0.6695331901, ||grad|| = 2.6378459339e+00, step = 1.000e+00
Iteration 31: f = 0.2959593162, ||grad|| = 1.5673611269e+01, step = 1.000e+00
Iteration 32: f = 0.1129875062, ||grad|| = 4.1627382894e+00, step = 1.000e+00
Iteration 33: f = 0.0216925886, ||grad|| = 5.0721938267e+00, step = 1.000e+00
Iteration 34: f = 0.0025859710, ||grad|| = 5.6070506769e-01, step = 1.000e+00
Iteration 35: f = 0.0000547515, ||grad|| = 3.0951128777e-01, step = 1.000e+00
Iteration 36: f = 0.0000007973, ||grad|| = 2.6396021184e-02, step = 1.000e+00
Iteration 37: f = 0.0000000002, ||grad|| = 3.8584198127e-04, step = 1.000e+00
Iteration 38: f = 0.0000000000, ||grad|| = 1.8034042174e-05, step = 1.000e+00
Iteration 39: f = 0.0000000000, ||grad|| = 8.2477097581e-09, step = 1.000e+00
"""
pattern = re.compile(r"Iteration\s+(\d+):\s+f\s*=\s*([0-9eE\.\-+]+)")

iterations = []
objective_values = []

for line in log.strip().split("\n"):
    match = pattern.search(line)
    if match:
        iterations.append(int(match.group(1)))
        objective_values.append(float(match.group(2)))

if not iterations:
    raise ValueError("No valid iteration logs found. Check your log format!")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, objective_values, marker="o", linestyle="-", color="tab:blue")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("C L-BFGS Convergence on Rosenbrock")
plt.grid(True)

# Optional: Save the plot
plt.savefig("c_lbfgs_convergence.png", dpi=300)
print("Convergence plot saved as c_lbfgs_convergence.png")
