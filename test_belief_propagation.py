import numpy as np

""" Define the simple tensor network:
###     x1=0 ---2--- x2=0
###      |             |
###      1             1
###      |             |
###     x4=1 ---2--- x3=1
###
Matching neighbors = 2, different neighbors = 1
Weight = 2 * 1 * 2 * 1 = 4
"""
F = np.array([
    [2, 1],
    [1, 2]
]) # rank 2 tensor (just a matrix)

Z = 0
for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            for x4 in range(2):

                weight = (
                    F[x1, x2]
                    * F[x2, x3]
                    * F[x3, x4]
                    * F[x4, x1]
                )

                Z += weight

print("Explicit Z:", Z)

tensors = [F, F, F, F]
network = "ab,bc,cd,da->"

""" Z is the partition function: the sum of the weights of all configurations.
Each weight is the product of the four factors F around the square, used as:
###     P(x) = weight(x) / Z
"""
Z = np.einsum(network, *tensors)
print("Z from network:", Z)

""" Belief propagation:
Neighboring nodes exchange information through their shared indices over messages.
For a single connection between two variables: x1 -- x2
###     For the contribution of the connection from x1 to x2:
###         Z = message_{1 -> 2} (0) + message_{1 -> 2} (1)
###     (0) means assuming x2=0 and (1) means assuming x2=1
###     where:
###         message_{1 -> 2} (0) = sum(F(x1, 0))
"""
m_0 = sum(F[x1, 0] for x1 in range(2))
print("m_1to2_at0: ", m_0)
m_1 = sum(F[x1, 1] for x1 in range(2))
print("m_1to2_at1: ", m_1)
Z_1to2 = m_0 + m_1
print("Z for two variables with one connection: ", Z_1to2)

""" Belief propagation:
A BP message tells a neighboring variable how much the entire network on the sending
side contributes for each value that neighbor could take.
Considering a chain of three connected variables: x1 -- x2 -- x3
###     For the contribution of the connection from x1 to x2:
###         Z = message_{1 -> 2} (0) + message_{1 -> 2} (1)
###     For the contribution of the connection from x2 to x3:
###         Z = message_{2 -> 3} (0) + message_{2 -> 3} (1)
###     where:
###         message_{2 -> 3} (0) = message_{1 -> 2}(0)*F(0,0) + message_{1 -> 2}(1)*F(1,0)
"""
m_1to2 = [
    sum(F[x1, x2] for x1 in range(2))
    for x2 in range(2)
]
m_2to3 = [
    sum(m_1to2[x2] * F[x2, x3] for x2 in range(2))
    for x3 in range(2)
]
Z = sum(m_2to3)

print("m_1to2:", m_1to2)
print("m_2to3:", m_2to3)
print("Z for a chain of three connected variables: ", Z)
