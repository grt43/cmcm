from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import scipy.optimize as opt


''' Compute the constraints in terms of a linear system.
Arguments:
    slab_width: Width in inches of a slab.
    slab_length: Length in inches of a slab.
    H: A of slabs to length in inches of each corner.
        Note that we label corners a,b,c,d with a,b being 
        the side that is encountered first and c,d being the latter
        with the slab being defined by a->b->d->c.
Returns:
    Matrix A and vector b such that Ax <= b satisfies constraints.
'''
def get_constraints(slab_width, slab_length, H):
    a,b,c,d = [0,1,2,3]
    n_corners = 4
    n_slabs = len(H)

    # Generate linear b to maintain complience with ADA requirements. 
    # Vertical difference between slabs cannot exceed 1/2 inch.
    vdiff_A = np.zeros((n_corners*(n_slabs-1), n_corners*n_slabs))
    vdiff_b = np.zeros(n_corners*(n_slabs-1))
    for i in range(n_slabs-1):
        slab = n_corners*i
        next_slab = n_corners*(i+1)

        # Ensure 'c' corner of current slab is within 
        # bounds of 'a' corner of next slab.
        vdiff_A[n_corners*i][slab + c] = 1
        vdiff_A[n_corners*i][next_slab + a] = -1

        vdiff_A[n_corners*i+1][slab + c] = -1
        vdiff_A[n_corners*i+1][next_slab + a] = 1

        vdiff_b[n_corners*i] = 0.5 + (H[i+1][a] - H[i][c])
        vdiff_b[n_corners*i+1] = 0.5 + (H[i][c] - H[i+1][a])

        # Ensure 'd' corner of current slab is within 
        # bounds of 'b' corner of next slab.
        vdiff_A[n_corners*i+2][slab + d] = 1
        vdiff_A[n_corners*i+2][next_slab + b] = -1

        vdiff_A[n_corners*i+3][slab + d] = -1
        vdiff_A[n_corners*i+3][next_slab + b] = 1

        vdiff_b[n_corners*i+2] = 0.5 + (H[i+1][b] - H[i][d])
        vdiff_b[n_corners*i+3] = 0.5 + (H[i][d] - H[i+1][b])

    # Angle of corners of slab in direction perpendicular 
    # to the road must be within 1-2 degrees.
    perp_A = np.zeros((2*n_slabs, n_corners*n_slabs))
    perp_b = np.zeros(2*n_slabs)
    for i in range(n_slabs):
        slab = n_corners*i
        perp_A[2*i][slab + a] = -1
        perp_A[2*i][slab + b] = 1

        perp_A[2*i+1][slab + a] = 1
        perp_A[2*i+1][slab + b] = -1

        perp_b[2*i] = -(slab_width * np.sin(np.pi * 1 / 180) + H[i][b] - H[i][a])
        perp_b[2*i+1] = slab_width * np.sin(np.pi * 2 / 180) + H[i][b] - H[i][a]

    # Angle of corners of slab in direction parallel 
    # to the road must be within -+2 degrees.
    pll_A = np.zeros((2*n_slabs, n_corners*n_slabs))
    pll_b = np.zeros(2*n_slabs)
    for i in range(n_slabs):
        slab = n_corners*i
        pll_A[2*i][slab + a] = 1
        pll_A[2*i][slab + c] = -1

        pll_A[2*i+1][slab + a] = -1
        pll_A[2*i+1][slab + c] = 1

        pll_b[2*i] = slab_length * np.sin(np.pi * 2 / 180) + H[i][c] - H[i][a]
        pll_b[2*i+1] = slab_length * np.sin(np.pi * 2 / 180) + H[i][a] - H[i][c]

    # Now need to compute contraints to ensure slab is still planar.
    planar_A = np.zeros((2*n_slabs, n_corners*n_slabs))
    planar_b = np.zeros(2*n_slabs)
    for i in range(n_slabs):
        slab = n_corners*i
        planar_A[2*i][slab + a] = 1
        planar_A[2*i][slab + b] = -1
        planar_A[2*i][slab + c] = -1
        planar_A[2*i][slab + d] = 1

        planar_A[2*i+1][slab + a] = -1
        planar_A[2*i+1][slab + b] = 1
        planar_A[2*i+1][slab + c] = 1
        planar_A[2*i+1][slab + d] = -1

        planar_b[2*i] = H[i][a] - H[i][b] - H[i][c] + H[i][d]
        planar_b[2*i+1] = H[i][a] - H[i][b] - H[i][c] + H[i][d]

    # Lastly, need to ensure that the delta_H does not push H below 0.
    zero_A = np.zeros((n_corners*n_slabs, n_corners*n_slabs))
    zero_b = np.zeros(n_corners*n_slabs)
    for i in range(n_slabs):
        slab = n_corners*i
        zero_A[slab + a][slab + a] = -1
        zero_A[slab + b][slab + b] = -1
        zero_A[slab + c][slab + c] = -1
        zero_A[slab + d][slab + d] = -1

        zero_b[slab + a] = H[i][a]
        zero_b[slab + b] = H[i][b]
        zero_b[slab + c] = H[i][c]
        zero_b[slab + d] = H[i][d]

    A = np.concatenate((vdiff_A, perp_A, pll_A, planar_A, zero_A), axis=0)
    b = np.concatenate((vdiff_b, perp_b, pll_b, planar_b, zero_b), axis=0)
    return A, b


''' Find the optimal repairs with repsect to cost.
Arguments:
    slab_width: Width in inches of a slab.
    slab_length: Length in inches of a slab.
    raise_weight: Weight with which we should choose to raise a slab.
    vut_weight: Weight with which we should choose to cut a slab.
    H: A of slabs to length in inches of each corner.
        Note that we label corners a,b,c,d with a,b being 
        the side that is encountered first and c,d being the latter
        with the slab being defined by a->b->d->c.
 '''
def optimal_repairs(slab_width, slab_length, raise_weight, cut_weight, H):
    a,b,c,d = [0,1,2,3]
    n_corners = 4
    n_slabs = len(H)

    # Vector representing the change in length of each 
    # corner of a given slab.
    delta_H = cp.Variable(n_corners * n_slabs)

    A, b = get_constraints(slab_width, slab_length, H)

    constraints = [A@delta_H <= b]

    objective = \
        cp.Minimize(
            cp.sum(
                raise_weight * cp.pos(delta_H) +
                cut_weight * cp.neg(delta_H)
            ) +
            cp.norm(delta_H)
        )
    prob = cp.Problem(objective, constraints)

    return delta_H, prob


''' Compute the real cost of the proposed changes.
'''
def compute_cost(raise_cost, cut_cost, replace_cost, deltas):
    cost = 0
    for slab in deltas:
        da, db, dc, dd = slab

        replaced = False

        dmax = np.max(slab)
        dmin = np.min(slab)

        if (dmin < 0 and dmin > -2):
            cost += cut_cost

        elif(dmin < 0):
            cost += replace_cost
            replaced = True

        if (dmax > 0) and (replaced == False):
            cost += raise_cost

    return cost


def main():
    # Height and width in inches.
    slab_width = 4
    slab_length = 5
    n_slabs = 25

    # Compute a random sidewalk.
    H = {}
    for i in range(n_slabs):
        a = 2 * np.random.rand() + 1
        b = a + np.random.rand() - 0.5
        c = a + np.random.rand() - 0.5
        d = b + c - a
        H[i] = [a, b, c, d]

    # H = {i:[0, 0, 0, 0] for i in range(n_slabs)}
    # H[int(n_slabs/2)] = [10,10,10,10]

    # Compute optimal solution.
    raise_cost, cut_cost, replace_cost = [5.16, 16.00, 22.00]

    raise_weight = raise_cost
    cut_weight = (cut_cost + replace_cost) / 2

    delta_H, prob = optimal_repairs(
        slab_width, slab_length, 
        raise_weight, cut_weight, H)
    prob.solve()
    deltas = delta_H.value.reshape((n_slabs, 4))

    # Compute cost of changes.
    cost = slab_width * slab_length * compute_cost(
        raise_cost, 
        cut_cost, 
        replace_cost, 
        deltas)

    np.set_printoptions(precision=2)
    print("Cost of changes = $%.2f" % cost)
    print("Changes: ")    
    print(deltas)

    # Plot results.
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim([-slab_width, 3*slab_width])
    ax.set_ylim([0, (n_slabs+1)*slab_length])
    ax.set_zlim([0, 8])
    for i in range(n_slabs):
        a, b, c, d = H[i]
        da, db, dc, dd = deltas[i]

        x_start = i*slab_width
        y_start = i*slab_length

        cmap_before = [1, 0, 0, 0.75]
        cmap_after = [0, 0, 1, 0.75]
        ax.plot_trisurf(
            [0, slab_width, 0, slab_width], 
            [y_start, y_start, y_start + slab_length, y_start + slab_length], 
            [a, b, c, d], 
            cmap=colors.ListedColormap(cmap_before))

        ax.plot_trisurf(
            [0 + slab_width, slab_width + slab_width, 0 + slab_width, slab_width + slab_width], 
            [y_start, y_start, y_start + slab_length, y_start + slab_length], 
            [a + da, b + db, c + dc, d + dd], 
            cmap=colors.ListedColormap(cmap_after))

    custom_lines = [Line2D([0], [0], color=cmap_before, lw=5),
                    Line2D([0], [0], color=cmap_after, lw=5)]
    ax.legend(custom_lines, ['Original sidewalk', 'After repairs'])
    #ax.view_init(elev=0, azim=0)
    plt.show()

if __name__ == "__main__":
    main()

