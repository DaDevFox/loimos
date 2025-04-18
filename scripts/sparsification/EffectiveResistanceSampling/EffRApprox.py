import cupy as cp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from tqdm import tqdm
from cupy.linalg import norm
from cupyx.scipy.sparse import diags, dia_matrix
from cupyx.scipy.sparse import coo_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import cg

def cg_gpu(A, b, x0=None, tol=1e-5, maxiter=None):
    m = A.shape[0]
        
    # Initial guess
    if x0 is None:
        x0 = cp.zeros(m, dtype=b.dtype)
    # Set maximum iterations
    if maxiter is None:
        maxiter = m
    x = x0
    Ax = cp.asarray(A.dot(x))
    r = b - Ax  # Residual
    p = r.copy()  # Initial search direction
    r_dot_r = cp.dot(r, r)
    for i in range(maxiter):
        Ap = cp.asarray(A.dot(p))  # Apply A to the direction vector
        alpha = r_dot_r / cp.dot(p, Ap)  # Step size
        
        # In-place updates to x, r, and p to save memory
        x += alpha * p  # Update the solution
        r -= alpha * Ap  # Update the residual
        r_dot_r_new = cp.dot(r, r)
        if cp.sqrt(r_dot_r_new) < tol:
            print(f"Convergence reached after {i+1} iterations.")
            break

        # In-place updates for p and the direction vector
        beta = r_dot_r_new / r_dot_r  # Update the search direction coefficient
        p *= beta  # In-place scaling of p
        p += r  # In-place addition to p
                
        r_dot_r = r_dot_r_new

    return x

def gpu_conjugate_gradient(A, b, x0=None, tol=1e-6, maxiter=1000):
    """Manual GPU-based Conjugate Gradient Solver using CuPy."""
    N = A.shape[0]
    if x0 is None:
        x = cp.zeros(N, dtype=cp.float32)
    else:
        x = x0

    r = b - A @ x
    p = r.copy()
    rs_old = cp.dot(r, r)

    for _ in range(maxiter):
        Ap = A @ p
        alpha = rs_old / (cp.dot(p, Ap) + 1e-10)
        x += alpha * p
        r -= alpha * Ap
        rs_new = cp.dot(r, r)

        if cp.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


# Transform adj matrix to edge list
# Par:
# adj; adj matrix
### Optimize!
def Mtrx_Elist(A):
    j, i = np.nonzero(np.triu(A))  # Find edges
    elist = np.vstack((i, j))
    weights = A[np.triu(A) != 0]  # Find weights

    return elist.transpose(), weights

def Mtrx_Elist_cp(A):
    # Find edges using CuPy
    j, i = cp.nonzero(cp.triu(A))  # CuPy equivalent of np.nonzero
    elist = cp.vstack((i, j))  # CuPy equivalent of np.vstack
    weights = A[cp.triu(A) != 0]  # CuPy equivalent of np.triu and indexing

    return elist.transpose(), weights

def Mtrx_Elist_cp_sparse(A):
    A = A.tocoo() # For easy row/col access

    # Find edges using CuPy
    mask = A.row < A.col
    i = A.row[mask]
    j = A.col[mask]
    weights = A.data[mask]

    elist = cp.vstack((i,j))

    return elist.transpose(), weights




# Legacy code
# def Mtrx_Elist(adj):
#     n = len(adj)
#     elist = []
#     weights = []
#     u_adj = np.triu(adj)
#     for j in range(n):
#         for i in range(n):
#             if u_adj[i][j] > 0:
#                 elist.append([j, i])
#                 weights.append(u_adj[i, j])
#     return np.array(elist), weights


# Transform edge list to adj matrix
# Par:
## E_list; edge list
## weights; edge weights
### Make sparse adj matrix for future?
def Elist_Mtrx(E_list, weights):
    n = np.max(E_list) + 1  # +1 for Python 0-index
    A = np.zeros(shape=(n, n))

    for i in range(np.shape(E_list)[0]):
        n1, n2 = E_list[i, :]
        w = weights[i]
        A[n1, n2], A[n2, n1] = w, w

    return A


# Transform edge list to sparse adj matrix
# Par:
## E_list; edge list
## weights; edge weights
### Make sparse adj matrix for future?
def Elist_Mtrx_s(E_list, weights):
    n = np.max(E_list) + 1  # +1 for Python 0-index
    A = sparse.csr_matrix((weights, (E_list[:, 0], E_list[:, 1])), shape=(n, n))
    A = A + A.transpose()

    return A

def Elist_Mtrx_s_cp(E_cp, weights_cp):
    """Build symmetric (undirected) adjacency matrix using CuPy and CSR format"""
    n = int(cp.max(E_cp)) + 1

    # Concatenate both directions for undirected graph
    row = cp.concatenate((E_cp[:, 0], E_cp[:, 1]))
    col = cp.concatenate((E_cp[:, 1], E_cp[:, 0]))
    data = cp.concatenate((weights_cp, weights_cp))

    # Create CSR matrix directly (auto-sums duplicate entries if any)
    A = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    return A



# Compute Laplacian, L
# Par:
## A; adj matrix
def Lap(A):
    L = np.diag(np.sum(abs(A), 1)) - A
    return L


# Compute Laplacian, L
# Par:
## A; sparse adj matrix
def Lap_s(A):
    L = sparse.csgraph.laplacian(A)
    return L

def Lap_s_cp(A):
    """
    Compute the combinatorial Laplacian L = D - A
    Assumes A is a CuPy sparse matrix (csr or coo)
    """
    # Compute degree vector: sum of weights per row
    degrees = cp.asarray(A.sum(axis=1)).ravel()
    # Create diagonal degree matrix
    D = diags(degrees)
    # Return Laplacian
    L = D - A
    return L

# Compute signed-edge vertex incidence matrix, B
# Par:
## E_list; edge list
def sVIM(E_list):
    m = np.shape(E_list)[0]  # number of edges
    E_list = E_list.transpose()  # make rows edge list

    data = [1] * m + [-1] * m  # arbitrary tails and heads
    i = list(range(0, m)) + list(range(0, m))  # i-th positions
    j = E_list[0, :].tolist() + E_list[1, :].tolist()  # j-th positions

    B = sparse.csr_matrix((data, (i, j)))  # Using sparse row matrix format for later use

    return B

def sVIM_cp(E_list_cp):
    """
    Compute signed vertex-incidence matrix B using CuPy.
    Assumes E_list_cp is a (m, 2) CuPy array of edges (tail, head).
    Returns a sparse CSR matrix of shape (m, n_nodes)
    """
    m = E_list_cp.shape[0]
    u = E_list_cp[:, 0]  # tails
    v = E_list_cp[:, 1]  # heads
    # Combine tail (+1) and head (-1) contributions
    data = cp.concatenate((cp.ones(m), -cp.ones(m)))
    row = cp.concatenate((cp.arange(m), cp.arange(m)))
    col = cp.concatenate((u, v))
    n_nodes = int(cp.max(E_list_cp)) + 1  # assume nodes are 0-indexed
    B = csr_matrix((data, (row, col)), shape=(m, n_nodes))
    return B


# Compute weights matrix, W
# Par:
## weights; edge weights
def WDiag(weights):
    m = len(weights)

    weights_sqrt = np.sqrt(weights)  # element-wise sqrt of weights for later use
    W = sparse.dia_matrix((weights_sqrt, [0]), shape=(m, m))  # Use more efficient dia sparse matrix

    return W

def WDiag_cp(weights_cp):
    """
    Compute the diagonal weights matrix W = diag(sqrt(weights))
    Assumes weights_cp is a CuPy array of edge weights
    """
    m = len(weights_cp)
    # Element-wise sqrt of weights using CuPy
    weights_sqrt = cp.sqrt(weights_cp)
    # Create diagonal sparse matrix (CuPy)
    W = dia_matrix((weights_sqrt, [0]), shape=(m, m))
    return W


# EffR Approximation
# method from Koutis et al.
# Par:
## E_list; edge list
## weights; list of weights
## epsilon; controls accuracy of approximation, increases computation time
## type; type of calculation for EffR
##
#### 'ext', exact calculation
#### 'ssa', original Spielman-Srivastava algorithm
#### 'kts', Koutis et. al
##### Implement preconditioner M for cg solver? cg(A,b,tol,M=None) - use spilu function or another from scipy.sparse.linalg? https://stackoverflow.com/questions/32865832/preconditioned-conjugate-gradient-and-linearoperator-in-python
##### !Warning! For very small networks, a preconditioner is advised!
def EffR(E_list, weights, epsilon, type, tol=1e-10, precon=False):
    # Find number of edges and number of nodes
    m = np.shape(E_list)[0]
    n = np.max(E_list) + 1

    # Obtain necessary matrices from edge list and edge weights
    A = Elist_Mtrx_s_cp(E_list, weights)  # adj matrix - sparse
    L = Lap_s_cp(A)  # Laplacian (sparse array)
    B = sVIM_cp(E_list)  # vertex indices matrix (crs)
    W = WDiag_cp(weights)  # Diagonal weight matrix (dia)
    scale = np.ceil(np.log2(n)) / epsilon  # set scale/resolution for Johnson-Lindenstrauss projection

    # Find preconditioner for L if precon is True
    if precon:
        M_inverse = sparse.linalg.spilu(L)
        M = sparse.linalg.LinearOperator((n, n), M_inverse.solve)

    # Ignore preconditioner if precon is False
    elif not precon:
        M = None

    # If preconditioner is passed, set M to precon
    else:
        M = precon

    # Exact effR values
    if type == 'ext':
        effR = np.zeros(shape=(1, m))
        if M is None:  # If no preconditioner
            for i in tqdm(range(m), desc="EffR"):
                Br = B[i, :].toarray()
                Z = gpu_conjugate_gradient(L, Br.transpose(), tol=tol)[0]
                R_eff = Br @ Z
                effR[:, i] = R_eff[0]
        else:  # If preconditioner
            for i in tqdm(range(m), desc="EffR"):
                Br = B[i, :].toarray()
                Z = gpu_conjugate_gradient(L, Br.transpose(), tol=tol, M=M)[0]
                R_eff = Br @ Z
                effR[:, i] = R_eff[0]

        effR = effR[0]
        return effR

    # Original Spielman-Srivastava algorithm
    if type == 'spl':

        # Define Q in type coo sparse matrix
        Q1 = sparse.random(int(scale), m, 1, format='csr') > 0.5
        Q2 = sparse.random(int(scale), m, 1, format='csr') > 0
        Q_not = Q1 - Q2  # need this to pass by invalid 'not' operator
        Q = Q1 + (-1 * Q_not)  # create Q matrix of 1s and -1s
        Q = Q / np.sqrt(scale)

        SYS = Q @ W @ B  # create system for Johnson-Lindenstrauss projection
        Z = np.zeros(shape=(int(scale), n))  # Create Z matrix to solve smaller dim SYS for effR

        if M is None:  # If no preconditioner
            for i in tqdm(range(int(scale)), desc="EffR"):
                SYSr = SYS[i, :].toarray()
                Z[i, :] = gpu_conjugate_gradient(L, SYSr.transpose(), tol=tol)[0]
        else:  # If preconditioner
            for i in tqdm(range(int(scale)), desc="EffR"):
                SYSr = SYS[i, :].toarray()
                Z[i, :] = gpu_conjugate_gradient(L, SYSr.transpose(), tol=tol, M=M)[0]

        effR = np.sum(np.square(Z[:, E_list[:, 0]] - Z[:, E_list[:, 1]]),
                      axis=0)  # Calculate distance between poitns for effR
        return effR

    # Koutis et al. algorithm
    if type == 'kts':
        effR_res = cp.zeros(shape=(1, m))

        WB = W.dot(B)

        if M is None:
            for i in tqdm(range(int(scale)), desc="EffR"):
                ons1_data = cp.random.rand(m) > 0.5  # Random binary data
                ons2_data = cp.random.rand(m) > 0  # Random binary data
                ons1 = csr_matrix((ons1_data.astype(cp.float32), (cp.zeros(m), cp.arange(m))), shape=(1, m))
                ons2 = csr_matrix((ons2_data.astype(cp.float32), (cp.zeros(m), cp.arange(m))), shape=(1, m))

                ons_not = ons1 - ons2  # need this to pass by invalid 'not' operator
                ons = ons1 + (-1 * ons_not)  # create Q matrix of 1s and -1s
                ons = ons / cp.sqrt(scale)

                #b = ons @ W @ B
                b = ons.dot(WB)

                Z, info = cg(L, b.toarray().T, tol=tol)
                Z = Z.T

                #effR_res = effR_res + np.abs(np.square(Z[E_list[:, 0]] - Z[E_list[:, 1]]))
                #effR_res = cp.sum(cp.abs(cp.square(Z[E_list[:, 0]] - Z[E_list[:, 1]])))
                effR_res = effR_res + cp.abs(cp.square(Z[E_list[:, 0]] - Z[E_list[:, 1]]))

        else:
            for i in tqdm(range(int(scale)), desc="EffR"):
                # Create memory saving vectors
                ons1 = sparse.random(1, m, 1, format='csr') > 0.5
                ons2 = sparse.random(1, m, 1, format='csr') > 0
                ons_not = ons1 - ons2  # need this to pass by invalid 'not' operator
                ons = ons1 + (-1 * ons_not)  # create Q matrix of 1s and -1s
                ons = ons / np.sqrt(scale)

                b = ons @ W @ B

                Z = gpu_conjugate_gradient(L, b.transpose(), tol=tol, M=M)[0]
                Z = Z.transpose()

                effR_res = effR_res + np.abs(np.square(Z[E_list[:, 0]] - Z[E_list[:, 1]]))

        effR = effR_res[0]
        return effR
