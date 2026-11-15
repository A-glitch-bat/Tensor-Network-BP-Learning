#--------------------------------

# Imports
import numpy as np
from itertools import product
#--------------------------------

""" Belief propagation:
Neighboring nodes exchange information through their shared indices over messages.
For a node i and a neighboring node j, the message m_{i -> j}(x_j) tells us how much the
rest of the network contributes when x_j takes each possible value.
The exact contraction is the reference result we compare against.
"""

def exact_partition_function(n, edges, F):
    """ Enumerate all 2^n configurations and sum their weights.
    For each assignment, every edge contributes a local factor F[x_i, x_j], and the total
    weight is the product of all such factors.
    """
    Z = 0.0

    for state in product(range(2), repeat=n):
        weight = 1.0

        for i, j in edges:
            weight *= F[state[i], state[j]]

        Z += weight

    return Z
#--------------------------------

""" A belief propagation  message is built by multiplying all incoming contributions to a node, 
excluding the neighbor that receives the message, and then summing over the sending variable.
"""


def build_neighbors(n, edges):
    neighbors = {i: [] for i in range(n)}

    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    return neighbors
# (this is the same idea as contracting a tensor network while keeping one index explicit)

def initialize_messages(edges):
    """ Start with a flat message for every directed edge.
    Each message is a length-2 vector for x = 0 and x = 1.
    """
    messages = {}

    for i, j in edges:
        messages[(i, j)] = np.array([0.5, 0.5])
        messages[(j, i)] = np.array([0.5, 0.5])

    return messages
#--------------------------------

def update_messages(n, edges, F, messages, neighbors):
    """ Update each directed message using the incoming messages from other neighbors.
    For message i -> j:
        m_{i->j}(x_j) = sum_{x_i} F(x_i, x_j) * prod_{k != j} m_{k->i}(x_i)
    """
    new_messages = {}

    for i, j in messages:
        incoming = np.ones(2)

        for k in neighbors[i]:
            if k != j:
                incoming *= messages[(k, i)]

        new_message = incoming @ F
        new_message /= np.sum(new_message)

        new_messages[(i, j)] = new_message

    return new_messages
#--------------------------------

def run_bp(n, edges, F, tolerance=1e-10, max_iter=1000):
    """ Iterate until the messages stop changing.
    This is still a toy example, but it follows the repeated message-passing pattern that
    appears in belief propagation.
    """
    neighbors = build_neighbors(n, edges)
    messages = initialize_messages(edges)

    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        new_messages = update_messages(n, edges, F, messages, neighbors)

        max_change = max(
            np.max(np.abs(new_messages[key] - messages[key]))
            for key in messages
        )

        messages = new_messages
        iterations = iteration

        if max_change < tolerance:
            converged = True
            break

    return messages, converged, iterations
#--------------------------------

def bethe_partition_function(n, edges, F, messages):
    """ Approximate the partition function using a Bethe-style factorization.
    This is just a compact way of combining the messages into an estimate for Z.
    """
    neighbors = build_neighbors(n, edges)

    Z_bethe = 1.0

    for i, j in edges:
        left = np.ones(2)
        right = np.ones(2)

        for k in neighbors[i]:
            if k != j:
                left *= messages[(k, i)]

        for k in neighbors[j]:
            if k != i:
                right *= messages[(k, j)]

        Z_ij = np.sum(F * np.outer(left, right))
        Z_bethe *= Z_ij

    for i in range(n):
        incoming = np.ones(2)

        for k in neighbors[i]:
            incoming *= messages[(k, i)]

        Z_i = np.sum(incoming)
        Z_bethe /= Z_i ** (len(neighbors[i]) - 1)

    return Z_bethe
#--------------------------------
