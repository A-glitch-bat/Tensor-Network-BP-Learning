#--------------------------------

# Imports
import numpy as np
import random
from itertools import product
#--------------------------------

""" Belief propagation:
Neighboring nodes exchange information through their shared indices over messages.
For a node i and a neighboring node j, the message m_{i -> j}(x_j) tells us how much 
the rest of the network contributes when x_j takes each possible value.
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

def initialize_messages(edges, initType="uniform", seed=None):
    """ Start with a flat message for every directed edge.
    Each message is a length-2 vector for x = 0 and x = 1.
    """
    messages = {}
    rseed = np.random.default_rng(seed)

    for i, j in edges:
        for source, target in [(i, j), (j, i)]:

            if initType == "uniform":
                message = np.array([0.5, 0.5])
            elif initType == "random":
                message = rseed.random(2)
                # this needs explicit normalization now that it's random
                message /= np.sum(message)
            else:
                raise ValueError("Invalid message initialization selected. Fix your code.")
            
            messages[(source, target)] = message

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

def run_bp(n, edges, F, tolerance=1e-10, max_iter=1000,
           initType="uniform", seed=None):
    """ Iterate until the messages stop changing.
    This is still a toy example, but it follows the repeated message-passing pattern that
    appears in belief propagation.
    """
    neighbors = build_neighbors(n, edges)
    messages = initialize_messages(edges, initType=initType, seed=seed)

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
    (a compact way of combining the messages into an estimate for Z)
    ### Bethe free energy      F_B = U_B - H_B
    ###     Approximates the free energy of the entire network using local beliefs.
    ###     The partition function is then estimated as Z_B = exp(-F_B).
    ###
    ### Bethe average energy   U_B = -sum of expected log(F) over all edges
    ###     Measures how well neighboring variable configurations agree with
    ###     their interaction factors F, weighted by their pairwise beliefs.
    ###
    ### Bethe entropy          H_B = sum of edge entropies - sum of node corrections
    ###     Approximates the uncertainty of the entire network using local beliefs.
    ###     Each edge contributes its entropy, but shared nodes are counted
    ###     repeatedly, so their entropy is subtracted (degree - 1) times.
    ###
    ### Implementing the Bethe approximation to the partition function, expressed 
    ### using local normalization factors obtained from BP messages:
    ###     Z_Bethe = product(Z_ij) / product(Z_i ** (degree_i - 1))
    ###     Z_ij: local edge normalization, including incoming BP messages
    ###     Z_i: local node normalization, combining all incoming BP messages
    ###     Corrects repeated node contributions when combining edge factors.
    ###     Exact on trees; approximate on graphs with cycles.
    """
    neighbors = build_neighbors(n, edges)

    # Numerator for Z_Bethe: edge contributions
    Z_Bethe = 1.0
    for i, j in edges:
        left = np.ones(2)
        right = np.ones(2)

        for k in neighbors[i]:
            if k != j:
                # collect incoming messages to i, excluding j
                left *= messages[(k, i)]

        for k in neighbors[j]:
            if k != i:
                # collect incoming messages to j, excluding i
                right *= messages[(k, j)]

        # contract the edge using both contributions
        Z_ij = np.sum(F * np.outer(left, right))
        Z_Bethe *= Z_ij # multiply edge contributions

    # Denominator for Z_Bethe: node corrections
    for i in range(n):
        incoming = np.ones(2)

        for k in neighbors[i]:
            # combine all incoming messages to node i
            incoming *= messages[(k, i)]

        # sum over both states of node i
        Z_i = np.sum(incoming)
        Z_Bethe /= Z_i ** (len(neighbors[i]) - 1) # correct node overcounting

    return Z_Bethe
#--------------------------------
