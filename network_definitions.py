#--------------------------------

# Imports
import numpy as np
#--------------------------------

""" Define the local tensor for the toy networks.
###     x1 -- x2
###      |     |
###     x4 -- x3
###
The same 2x2 matrix is used on every edge, so each factor is:
F(x_i, x_j) = [[2, 1], [1, 2]]
This is a simple rank-2 tensor, and the total weight of a configuration is the product
of all edge factors around the graph.
"""
F = np.array([
    [2.0, 1.0],
    [1.0, 2.0]
])

""" Define a few small example networks.
###     Tree:        0 - 1 - 2
###                    \
###                     3
###
###     Single loop: 0 - 1
###                  |   |
###                  3 - 2
###
###     Multiple loops: 0 - 1
###                      | \ |
###                      3 - 2
###
Each graph is described by its number of variables and its undirected edges.
"""
networks = {
    "Tree": {
        "n": 4,
        "edges": [(0, 1), (1, 2), (1, 3)]
    },
    "Single loop": {
        "n": 4,
        "edges": [(0, 1), (1, 2), (2, 3), (3, 0)]
    },
    "Multiple loops": {
        "n": 4,
        "edges": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    }
}
#--------------------------------
