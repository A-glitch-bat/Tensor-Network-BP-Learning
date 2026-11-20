#--------------------------------

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from network_generators import make_grid
from belief_propagation import (
    build_neighbors,
    initialize_messages,
    update_messages,
)
#--------------------------------

networks = {
    "Grid 3x3": make_grid(3, 3),
    "Grid 4x4": make_grid(4, 4),
}

strengths = [5.0, 10.0]
seeds = range(10)

tolerance = 1e-10
max_iter = 1000

results = []
trajectories = []
#--------------------------------

# Run BP while recording its trajectory to plot
def run_bp_tracked(n, edges, F, seed):
    neighbors = build_neighbors(n, edges)

    rng = np.random.default_rng(seed)
    messages = {}

    for i, j in edges:
        for key in [(i, j), (j, i)]:
            v = rng.random(2)
            messages[key] = v / v.sum()

    history = []
    converged = False

    for iteration in range(1, max_iter + 1):
        new_messages = update_messages(
            n, edges, F, messages, neighbors
        )

        max_change = max(
            np.max(np.abs(new_messages[key] - messages[key]))
            for key in messages
        )

        history.append(max_change)
        messages = new_messages

        if max_change < tolerance:
            converged = True
            break

    # Independently check the final BP fixed-point residual
    next_messages = update_messages(
        n, edges, F, messages, neighbors
    )

    residual = max(
        np.max(np.abs(next_messages[key] - messages[key]))
        for key in messages
    )

    # Distance of messages from the symmetric solution
    asymmetry = np.mean([
        abs(message[1] - 0.5)
        for message in messages.values()
    ])

    # Determine the dominant state across messages
    mean_p1 = np.mean([
        message[1] for message in messages.values()
    ])

    if abs(mean_p1 - 0.5) < 1e-5:
        state = "Symmetric"
    elif mean_p1 > 0.5:
        state = "State 1"
    else:
        state = "State 0"

    return {
        "converged": converged,
        "iterations": iteration,
        "residual": residual,
        "asymmetry": asymmetry,
        "mean_p1": mean_p1,
        "state": state,
        "history": history,
    }
#--------------------------------


# Experiments
for name, network in networks.items():
    n = network["n"]
    edges = network["edges"]

    for s in strengths:
        F = np.array([
            [s, 1.0],
            [1.0, s]
        ])

        for seed in seeds:
            result = run_bp_tracked(n, edges, F, seed)

            results.append({
                "network": name,
                "s": s,
                "seed": seed,
                "converged": result["converged"],
                "iterations": result["iterations"],
                "residual": result["residual"],
                "asymmetry": result["asymmetry"],
                "mean_p1": result["mean_p1"],
                "state": result["state"],
            })

            for iteration, change in enumerate(
                result["history"], start=1
            ):
                trajectories.append({
                    "network": name,
                    "s": s,
                    "seed": seed,
                    "iteration": iteration,
                    "max_change": change,
                })
#--------------------------------

# Save and print
df = pd.DataFrame(results)
df.to_csv("BP_fixed_points.csv", index=False)

history_df = pd.DataFrame(trajectories)
history_df.to_csv("BP_trajectories.csv", index=False)

print("\nSTAGE 3 SUMMARY")
print("-" * 65)

summary = df.groupby(["network", "s"]).agg(
    convergence_rate=("converged", "mean"),
    mean_iterations=("iterations", "mean"),
    mean_residual=("residual", "mean"),
    mean_asymmetry=("asymmetry", "mean"),
)

print(summary.round(6))

print("\nFINAL MESSAGE STATES")
print(pd.crosstab(
    [df["network"], df["s"]],
    df["state"]
))

# Plot
for name in networks:
    for s in strengths:
        subset = history_df[
            (history_df["network"] == name) &
            (history_df["s"] == s)
        ]

        plt.figure(figsize=(9, 5))

        for seed in seeds:
            run = subset[subset["seed"] == seed]

            plt.plot(
                run["iteration"],
                run["max_change"],
                alpha=0.65,
                label=f"Seed {seed}"
            )

        plt.axhline(
            tolerance,
            linestyle="--",
            color="black",
            label="Tolerance"
        )

        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Maximum message change")
        plt.title(f"{name}, s={s}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        filename = (
            f"BP_stage3_{name.replace(' ', '_')}_s{s:g}.png"
        )

        plt.savefig(filename, dpi=150)
        plt.close()

print("\nResults and plots saved.")
#--------------------------------
