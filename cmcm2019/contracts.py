import numpy as np
import argparse

''' Performs the neighbor joining algorithm on a given set of sequences.
Arguments:
    D: map of maps, defining distances between the sequences
       (initially n x n, symmetric, 0 on the diagonal)
       (index -> index -> distance)
Returns:
    edges: edges formed using the neighbor-join algorithm
'''
def neighbor_join(d):
    n = len(d)
    L = list(range(n))
    edges = []

    # Define k to be outside name mapping.
    k = n

    while(len(L) > 2):
        # Compute averaged distances.
        D = {i: {j: 0 for j in L} for i in L}
        r = {i: 0 for i in range(k)}

        # First compute distance averages.
        for i in range(k):
            for j in L:
                r[i] +=  d[i][j] / (len(L) - 2)

        for i in L:
            for j in L:
                D[i][j] = d[i][j] - (r[i] + r[j])

        # Find pair i,j in L that minimizes Dij.
        i_min = L[0]
        j_min = L[1]
        min_val = D[i_min][j_min]
        for i in L:
            for j in L:
                if (i != j) and (D[i][j] < min_val):
                    min_val = D[i][j]
                    i_min = i
                    j_min = j

        # Add new node k and add edge.
        edges.append((i_min, k))
        edges.append((j_min, k))
        L.remove(i_min)
        L.remove(j_min)
        L.append(k)

        # Update distance dictionary to include k.
        d[k] = {i: 0 for i in range(k + 1)}

        for m in d[k]:
            d[k][m] = (1 / 2) * (d[i_min][m] + d[j_min][m] - d[i_min][j_min])
            d[m][k] = d[k][m]

        d[i_min][k] = (1 / 2) * (d[i_min][j_min] + r[i_min] - r[j_min])
        d[j_min][k] = d[i_min][j_min] - d[i_min][k]

        d[k][i_min] = d[i_min][k]
        d[k][j_min] = d[j_min][k]

        d[k][k] = 0

        k += 1

    # Add termination case.
    edges.append((L[0],L[1]))

    return edges, d


''' Helper function for defining a tree data structure.
    First finds the edge to add a root node to and then generates binary tree.
    Root node should be at the midpoint of the longest branch.
Arguments:
    n: number of vertices
    edges: edges in the tree
Returns:
    root: root node
    tree_map: map from nodes in the tree -> list of children (leaves have
              empty lists)
'''
def assemble_tree(root, edges):
    children = []
    new_edges = []
    for (i,j) in edges:
        if i == root:
            children.append(j)

        elif j == root:
            children.append(i)
            
        else:
            new_edges.append((i,j))

    tree_map = {root: children}
    for child in children:
        tree_map.update(assemble_tree(child, new_edges))

    return tree_map


''' Breadth first search.
Arguments:
    root: root node
    tree_map: map from nodes in the tree -> list of children (leaves have
              empty lists)
Returns:
    visited: list of vertices in order vistied
'''
def bfs(root, tree_map):
    visited = []
    queue = [root]
 
    while queue:
        node = queue.pop(0)
        if node not in visited:
            
            visited.append(node)
            neighbours = tree_map[node]
            
            for neighbour in neighbours:
                queue.append(neighbour)

    return visited


''' Returns a string of the Newick tree format for the tree rooted at `root`.

Arguments:
    root: root of the tree (int)
    tree_map: map from node to list, describing each node's immediate children
    D: distance matrix of all nodes
    mapping: index to name mapping (dictionary)
Returns:
    output: rooted tree in Newick tree format (string)
'''
def generate_newick(root, tree_map, d, mapping = None, first = True):
    if not tree_map[root]:
        if mapping != None:
            return mapping[root]
        else:
            return str(root)

    (i, j) = tree_map[root]

    i_length = (d[i][j] / 2) if first else d[root][i]
    j_length = (d[i][j] / 2) if first else d[root][j]


    output = "({}:{:.6f},{}:{:.6f})".format(
        generate_newick(i, tree_map, d, mapping, first = False),
        i_length,
        generate_newick(j, tree_map, d, mapping, first = False),
        j_length)

    if first:
        output += ";"

    return output



def main():
    n = 20
    points = [0 for i in range(n)]
    for i in range(n):
        points[i] = (np.random.randint(100) + 1, np.random.randint(100) + 1)

    d = {i:{} for i in range(n)}
    for i in range(n):
        for j in range(n):
            d[i][j] = np.sqrt((points[i][0] - points[j][0])**2 +(points[i][1] - points[j][1])**2)

    edges, d = neighbor_join(d)

    # Find maximum length edge.
    d_max = 0
    i_max = 0
    j_max = 0
    for (i, j) in edges:
        if d[i][j] > d_max:
            d_max = d[i][j]
            i_max = i
            j_max = j

    # Replace maximum edge with root connections.
    root = len(d)
    edges.remove((i_max, j_max))
    edges.append((i_max, root))
    edges.append((j_max, root))

    tree_map = assemble_tree(root, edges)
    print(generate_newick(root, tree_map, d))
    print(bfs(root, tree_map))


if __name__ == "__main__":
    main()