"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    # Initialising leaf nodes
    nodes = []
    for label in labels:
        nodes.append(Node(label, None, 0, None, 0))

    # Loop through neighbour joining algorithm until we reach convergence
    converge = False
    while (converge == False):
        # New array with the sum of each row
        add_distance = add_rows(distances, labels)
        # New array with Q values
        Q_table = table_Q(add_distance, labels)

        # Getting the index of the minimum value from Q table
        min_index = np.argmin(Q_table)
        row, col = np.unravel_index(min_index, Q_table.shape)
        
        taxon_1 = labels[row]
        taxon_2 = labels[col]

        # Creating a new label list based on joint taxons
        new_label = labels.copy()
        new_label.remove(taxon_1)
        new_label.remove(taxon_2)
        new_label.append(taxon_1 + taxon_2)

        # Calculating distance and appending nodes
        n = len(labels)
        node_1_idx = find_node_index(nodes, taxon_1)
        node_2_idx = find_node_index(nodes, taxon_2)
        if (n == 2):
            root = Node(taxon_1 + taxon_2, nodes[node_1_idx], distances[row][col]/2, nodes[node_2_idx], distances[row][col]/2)
            nodes.append(root)
            converge = True
        else:
            taxon_1_dis = 1/2*distances[row][col] + (1/(2*(n - 2))) * (add_distance[row][n] - add_distance[col][n])
            taxon_2_dis = distances[row][col] - taxon_1_dis
            nodes.append(Node(taxon_1 + taxon_2, nodes[node_1_idx], taxon_1_dis, nodes[node_2_idx], taxon_2_dis))

        # Updating distance matrix and labels for next iteration  
        distances = new_distance(add_distance, new_label, labels, row, col)
        labels = new_label
    
    # for node in nodes:
    #     print("Name:", node.name, end=', ')
    #     print("Left:", node.left, end=', ')
    #     print("Left distance:", node.left_distance, end=', ')
    #     print("Right:", node.right, end=', ')
    #     print("Right distance:", node.right_distance)
    return root

# Function that finds the index of a node when a list of nodes and the name
# searching for is passed in
def find_node_index(nodes, name):
    i = 0
    for node in nodes:
        if node.name == name:
            return i
        i += 1

# Function that takes in distance matrix and returns a table with an extra
# column that holds the sum of each row
def add_rows(distances, labels):
    n = len(labels)
    sum_column = []

    # Ensuring distance matrix is symmetric (assuming top half is filled)
    for i in range(n):
            for j in range(i + 1, n):
                distances[j][i] = distances[i][j]

    # Getting sum of each column and storing each value 
    for i in range(n):
        total = 0
        for j in range(n):
            total += distances[i][j]
        sum_column = np.append(sum_column, total)

    # Adding the extra column
    d_array = np.column_stack((distances, sum_column))

    return(d_array)

# Function that takes in a distance array with an extra column with the sum for
# each row, and returns a new table with Q values calculated
def table_Q(d_array, labels):
    n = len(labels)
    q_array = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            q_array[i][j] = (n-2) * d_array[i][j] - d_array[i][n] - d_array[j][n]
            q_array[j][i] = q_array[i][j]

    return(q_array)

# Function that takes in a distance array, new labels, old labels, and the
# row and column of minimum value from Q table. 
# A new distance table is calculated and returned
def new_distance(d_array, new_labels, old_labels, row, col):
    n = len(new_labels)

    new_distance = np.zeros((n, n))
    dist = d_array[row][col]

    for i in range(n - 1):
        old_index = old_labels.index(new_labels[i])

        new_distance[n - 1][i] = 1/2 * (d_array[row][old_index] + d_array[col][old_index] - dist)
        new_distance[i][n - 1] = new_distance[n - 1][i]

    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            old_i = old_labels.index(new_labels[i])
            old_j = old_labels.index(new_labels[j])

            new_distance[i][j] = d_array[old_i][old_j]
            new_distance[j][i] = new_distance[i][j]

    return(new_distance)

def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """
    if tree is None:
        return

    stack = [tree]

    while stack:
        curr_node = stack.pop()

        if curr_node.left:
            stack.append(curr_node.left)
            print(curr_node.name, curr_node.left_distance, curr_node.right_distance)

        if curr_node.right:
            stack.append(curr_node.right)
            print(curr_node.name, curr_node.left_distance, curr_node.right_distance)

        if curr_node.left is None and curr_node.right is None:
            print(f"LEAF:{curr_node.name}")
            
    return ax
    

def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    raise NotImplementedError()


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()