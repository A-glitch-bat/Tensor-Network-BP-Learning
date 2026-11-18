#--------------------------------

# Imports
import numpy as np
import pandas as pd
from network_definitions import networks
from belief_propagation import (
    exact_partition_function,
    run_bp,
    bethe_partition_function,
)
#--------------------------------

strengths = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
seeds = range(30)
results = []

for name, network in networks.items():
    print(name, network)
    n = network["n"]
    edges = network["edges"]

    for s in strengths:
        F = np.array([[s, 1.0], [1.0, s]])
        Z_exact = exact_partition_function(n, edges, F)

        for seed in seeds:
            messages, converged, iterations = run_bp(
                n, edges, F,
                initType="random",
                seed=seed
            )
            print(messages)

            Z_bethe = (
                bethe_partition_function(n, edges, F, messages)
                if converged else np.nan
            )

            results.append({
                "network": name,
                "s": s,
                "seed": seed,
                "Z_exact": Z_exact,
                "Z_bethe": Z_bethe,
                "relative_error": abs(Z_exact - Z_bethe) / Z_exact
                                  if converged else np.nan,
                "converged": converged,
                "iterations": iterations
            })
#--------------------------------

df = pd.DataFrame(results)
df.to_csv("BP_simple_graph_results.csv", index=False)
print(df)
#--------------------------------
