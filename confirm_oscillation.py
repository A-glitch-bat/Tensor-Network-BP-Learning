#--------------------------------

# Imports
import numpy as np
from network_generators import make_grid
from belief_propagation import (
    build_neighbors,
    initialize_messages,
    update_messages,
)
#--------------------------------


network = make_grid(3, 3)
n, edges = network["n"], network["edges"]
F = np.array([[5.0, 1.0], [1.0, 5.0]])

neighbors = build_neighbors(n, edges)
messages = initialize_messages(edges, initType="random", seed=1)
#--------------------------------


previous = None
for iteration in range(1, 1001):
    new = update_messages(n, edges, F, messages, neighbors)

    # Difference between consecutive iterations
    delta_1 = max(
        np.max(np.abs(new[k] - messages[k]))
        for k in messages
    )

    # Difference between iterations two steps apart
    delta_2 = (
        max(
            np.max(np.abs(new[k] - previous[k]))
            for k in messages
        )
        if previous is not None else np.nan
    )

    if iteration > 990:
        print(f"{iteration}: Δ1={delta_1:.3e}, Δ2={delta_2:.3e}")

    previous = messages
    messages = new
#--------------------------------
