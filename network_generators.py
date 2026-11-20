#--------------------------------
 
# Chain: 0 -- 1 -- 2 -- ... -- n-1
def make_chain(n):
    edges = [(i, i + 1) for i in range(n - 1)]
    return {"n": n, "edges": edges}

# Ring: chain with the ends connected
def make_ring(n):
    edges = [(i, i + 1) for i in range(n - 1)]
    edges.append((n - 1, 0))
    return {"n": n, "edges": edges}

# Binary tree: each parent has two children
def make_binary_tree(depth):
    n = 2 ** (depth + 1) - 1
    edges = []

    for i in range(n):
        for child in (2 * i + 1, 2 * i + 2):
            if child < n:
                edges.append((i, child))

    return {"n": n, "edges": edges}

# Grid: nodes connected horizontally and vertically
def make_grid(rows, cols):
    n = rows * cols
    edges = []

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c

            if c + 1 < cols:
                edges.append((i, i + 1))

            if r + 1 < rows:
                edges.append((i, i + cols))

    return {"n": n, "edges": edges}
#--------------------------------
