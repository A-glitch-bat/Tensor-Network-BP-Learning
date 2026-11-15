#--------------------------------

# Imports
from network_definitions import F, networks
from belief_propagation import (
    exact_partition_function,
    run_bp,
    bethe_partition_function,
)
#--------------------------------

""" Run the small tensor-network examples and compare the exact contraction to the
Belief Propagation estimate.
###     Exact Z: sum over all configurations
###     BP Z: estimate from iterated messages
The point is to check that the message updates behave as expected for tiny graphs.
"""

for name, network in networks.items():
    n = network["n"]
    edges = network["edges"]

    Z_exact = exact_partition_function(n, edges, F)
    messages, converged, iterations = run_bp(n, edges, F)
    Z_bethe = bethe_partition_function(n, edges, F, messages)

    relative_error = abs(Z_exact - Z_bethe) / Z_exact

    print(f"\n{name}")
    print("-" * 40)
    print("Edges:", edges)
    print("Exact Z:", Z_exact)
    print("Bethe Z:", Z_bethe)
    print("Relative error:", f"{relative_error:.6%}")
    print("Converged:", converged)
    print("Iterations:", iterations)
#--------------------------------
