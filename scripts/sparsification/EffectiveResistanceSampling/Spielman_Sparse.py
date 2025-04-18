import random as ran
import cupy as cp
from cupyx.scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import numpy as np
import numpy as np
import scipy.sparse as sparse
from EffRApprox import Mtrx_Elist
from tqdm import tqdm
import cupy as cp


# from virtualenvs.AdaptiveAlgo import Adapt1
# from virtualenvs.Spielman_EffR import Mtrx_Elist


# Normalize probs such that sum(probs)=1
# Input:
# P - list of probs
# Output:
# P_n - list of probs' that sum to 1
def normprobs(P):
    prob_fac = 1 / sum(P)
    P_n = [prob_fac * p for p in P]
    return np.array(P_n)

import cupy as cp

def normprobs_cp(P):
    # Normalize probs such that sum(probs) = 1 using CuPy
    prob_fac = 1 / cp.sum(P)  # Use cp.sum for summing elements on GPU
    P_n = prob_fac * P  # Element-wise multiplication
    return P_n  # Return as a CuPy array (already on GPU)


# Normalize probs such that sum(probs)=1
# Input:
# P - list of probs
# Output:
# P_n - list of probs' that sum to 1
def normprobs(P):
    prob_fac = 1 / sum(P)
    P_n = [prob_fac * p for p in P]
    return np.array(P_n)



# Create a list of edge R_eff
# Input:
# R - Array of R_effs
# adj - Adj matrix
# Output:
# R_list - list of edge R_eff
# NOT NEEDED WITH NEW EFFR CODE - MAY, 2021
# def EffR_List(R, adj):
#     R_list = []
#     adj = np.triu(adj)
#     R = np.triu(R)
#     for i in range(len(adj)):
#         for j in range(len(adj)):
#             if adj[i][j] > 0:
#                 R_list.append(R[i][j])
#     return R_list


# Create a effective resistance sparsifer
# From Spielman and Srivastava 2008
# Input:
# adj - Adj matrix
# q - number of samples
# R - Matrix of effective resistances (other types of edge importance in the future, possibly)
# Output:
# H - effective resistance sparsifer adj matrix
def Spl_EffRSparse(n, E_list, weights, q, effR, seed=None):
    ran.seed(seed)
    P = []
    for i in tqdm(range(len(E_list)), desc="Spl_EffRSparse_1"):
        w_e = weights[i]
        R_e = effR[i]
        P.append((w_e * R_e) / (n - 1))
    Pn = np.array(normprobs(P))
    C = ran.choices(list(zip(E_list, weights, Pn)), Pn, k=q)
    H = np.zeros(shape=(n, n))
    for x in tqdm(range(q), desc="Spl_EffRSparse_2"):
        e, w_e, p_e = C[x][0], C[x][1], C[x][2]
        H[e[0]][e[1]] += w_e / (q * p_e)
    return H + np.transpose(H)


# Create a effective resistance sparsifer with sparse matrix
# From Spielman and Srivastava 2008
# Input:
# adj - Adj matrix
# q - number of samples
# R - Matrix of effective resistances (other types of edge importance in the future, possibly)
# Output:
# H - effective resistance sparsifer adj matrix
def Spl_EffRSparse_s(n, E_list, weights, q, effR, seed=None):
    ran.seed(seed)
    P = []
    H_list = np.zeros((len(E_list),3))
    for i in range(len(E_list)):
        w_e = weights[i]
        R_e = effR[i]
        P.append((w_e * R_e) / (n - 1))
    Pn = np.array(normprobs(P))
    C = ran.choices(list(zip(E_list, weights, Pn, range(len(E_list)))), Pn, k=q)
    for x in range(q):
        e, w_e, p_e, i = C[x][0], C[x][1], C[x][2], C[x][3]
        H_list[i,2] += w_e / (q * p_e)
        H_list[i,0:2] = e[0], e[1]
    H = sparse.csr_matrix((H_list[:,2], (E_list[:,0], E_list[:,1])), shape=(n,n))
    return H + H.transpose()

def safe_cupy_array(py_list, dtype=cp.int32, chunk_size=100_000):
    """Convert a large Python list to a CuPy array without OOMing."""
    n = len(py_list)
    arr = cp.empty(n, dtype=dtype)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        arr[i:end] = cp.asarray(py_list[i:end], dtype=dtype)
    return arr


def Spl_EffRSparse_cp(n, E_list, weights, q, effR, seed=None):
    ran.seed(seed)
    P = []
    for i in tqdm(range(len(E_list)), desc="Spl_EffRSparse_1"):
       w_e = weights[i]
       R_e = effR[i]
       P.append((w_e * R_e) / (n - 1))

    # TODO: debug OOM error

    Pn = cp.array(normprobs_cp(cp.asarray(P)))
    C = ran.choices(
    list(zip(E_list, weights, Pn.get())), Pn.get(), k=q)  # CuPy to NumPy conversion for random.choices

    row_idx = cp.empty(q, dtype=cp.int32)
    col_idx = cp.empty(q, dtype=cp.int32)
    data = cp.empty(q, dtype=cp.float32)
    for x in tqdm(range(q), desc="Spl_EffRSparse_2"):
        e, w_e, p_e = C[x][0], C[x][1], C[x][2]
        value = w_e / (q * p_e)
        row_idx[x] = e[0]
        col_idx[x] = e[1]
        data[x] = value

    H_sparse = coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
    H_symm = H_sparse + H_sparse.T  # Symmetric sparse matrix
    return H_symm


def Spl_EffRSparse_cp_2(n, E_list, weights, q, effR, seed=None):
    if seed is not None:
        cp.random.seed(seed)

    # Move data to GPU
    weights_gpu = cp.asarray(weights)
    effR_gpu = cp.asarray(effR)
    # Vectorized probability calculation
    P_gpu = (weights_gpu * effR_gpu) / (n - 1)
    P_sum = cp.sum(P_gpu)
    Pn_gpu = P_gpu / P_sum
    # Batch sampling to reduce memory usage

    batch_size = min(1_000, q)
    sampled_rows = []
    sampled_cols = []
    sampled_data = []
    for batch_start in range(0, q, batch_size):
            batch_end = min(batch_start + batch_size, q)
            batch_q = batch_end - batch_start
            # Sample indices on GPU
            sampled_indices = cp.random.choice(len(E_list), size=batch_q, replace=True, p=Pn_gpu)
            # Count occurrences of each edge
            unique_indices, counts = cp.unique(sampled_indices, return_counts=True)
            # Retrieve edge data
            rows = cp.asarray(E_list)[unique_indices, 0]
            cols = cp.asarray(E_list)[unique_indices, 1]
            data = (
                    weights_gpu[unique_indices] * counts / (q * Pn_gpu[unique_indices])
                    )

    # Append to batch results
    sampled_rows.append(rows)
    sampled_cols.append(cols)
    sampled_data.append(data)
    # Concatenate all batches
    sampled_rows = cp.concatenate(sampled_rows)
    sampled_cols = cp.concatenate(sampled_cols)
    sampled_data = cp.concatenate(sampled_data)
    # Create sparse matrix
    H_sparse = coo_matrix((sampled_data, (sampled_rows, sampled_cols)), shape=(n, n))
    H_csr = H_sparse.tocsr()
    # Symmetrize the matrix
    H_symm = H_csr + H_csr.T
    return H_symm.get()


# Input:
# adj - Adj matrix
# q - number of samples
# Output:
# H
def UniSampleSparse(n, E_list, weights, q, seed=None):
    ran.seed(seed)
    Pn = [1 / len(E_list)] * len(E_list)
    C = ran.choices(list(zip(E_list, weights, Pn)), Pn, k=q)
    H = np.zeros(shape=(n, n))
    for x in range(q):
        e = C[x][0]
        w_e = C[x][1]
        p_e = C[x][2]
        H[e[0]][e[1]] += w_e / (q * p_e)
    return H + H.transpose()


# Create a random uniform sparsifier
# Input:
# adj - Adj matrix
# q - number of samples
# Output:
# H
def UniSampleSparse_s(n, E_list, weights, q, seed=None):
    ran.seed(seed)
    Pn = [1 / len(E_list)] * len(E_list)
    C = ran.choices(list(zip(E_list, weights, Pn, range(len(E_list)))), Pn, k=q)
    H_list = np.zeros((len(E_list),3))
    for x in range(q):
        e, w_e, p_e, i = C[x][0], C[x][1], C[x][2], C[x][3]
        H_list[i,2] += w_e / (q * p_e)
        H_list[i,0:2] = e[0], e[1]
    H = sparse.csr_matrix((H_list[:, 2], (E_list[:, 0], E_list[:, 1])), shape=(n, n))
    return H + H.transpose()


# Create a sparsifier with edge weights
# Input:
# adj - Adj matrix
# q - number of samples
# Output:
# H
def WeightSparse_s(n, E_list, weights, q, seed=None):
    ran.seed(seed)
    Pn = normprobs(weights)
    C = ran.choices(list(zip(E_list, weights, Pn, range(len(E_list)))), Pn, k=q)
    H_list = np.zeros((len(E_list),3))
    for x in range(q):
        e, w_e, p_e, i = C[x][0], C[x][1], C[x][2], C[x][3]
        H_list[i,2] += w_e / (q * p_e)
        H_list[i,0:2] = e[0], e[1]
    H = sparse.csr_matrix((H_list[:, 2], (E_list[:, 0], E_list[:, 1])), shape=(n, n))
    return H + np.transpose(H)


def Thresh(n, E_list, weights, per):
    m = int(np.ceil(per * len(weights)))
    n_weights = [0] * len(weights)
    weights = list(enumerate(weights))
    weights = sorted(weights, key=lambda tup: tup[1], reverse=True)
    weights = weights[0:m]
    for i in weights:
        n_weights[i[0]] = i[1]
    H = sparse.csr_matrix((n_weights, (E_list[:,0], E_list[:,1])), shape=(n,n))
    return H


# Create a S-S sparsifier with a specified number of edges.
# Input:
# adj - Adj matrix
# e - number of edges
# Output:
# H - effective resistance sparsifer adj matrix
def SSEdge(adj, R, e):
    H = np.zeros(shape=(len(adj), len(adj)))
    r_tick = int(1.25 * e)
    while len(Mtrx_Elist(H)[0]) != e:
        H = Spl_EffRSparse(adj, r_tick, R)
        if len(Mtrx_Elist(H)[0]) > e:
            r_tick = r_tick - (len(Mtrx_Elist(H)[0]) - e)
        if len(Mtrx_Elist(H)[0]) < e:
            r_tick = r_tick + (e - len(Mtrx_Elist(H)[0]))
        print(len(Mtrx_Elist(H)[0]))
    return H


# Create a Uni sparsifier with a specificed number of edges.
# Input:
# adj - Adj matrix
# e - number of edges
# Output:
# H - uni sparsifer adj matrix
def UniEdge(adj, e):
    H = np.zeros(shape=(len(adj), len(adj)))
    r_tick = int(0.9 * e)
    while len(Mtrx_Elist(H)[0]) != e:
        H = UniSampleSparse(adj, r_tick)
        if len(Mtrx_Elist(H)[0]) > e:
            r_tick = r_tick - (len(Mtrx_Elist(H)[0]) - e)
        if len(Mtrx_Elist(H)[0]) < e:
            r_tick = r_tick + (e - len(Mtrx_Elist(H)[0]))
        print(len(Mtrx_Elist(H)[0]))
    return H

# def AdaptEdge(adj, R, T, e):
#    H = np.zeros(shape=(len(adj), len(adj)))
#    r_tick = int(1.5 * e)
#    while len(Mtrx_Elist(H)[0]) != e:
#        H = Adapt1(adj, r_tick, R, T)
#        if len(Mtrx_Elist(H)[0]) > e:
#            r_tick = int(r_tick - (len(Mtrx_Elist(H)[0]) - (e * 1/T)))
#        if len(Mtrx_Elist(H)[0]) < e:
#            r_tick = int(r_tick + (e - (len(Mtrx_Elist(H)[0]) * (1/T))))
#        print(len(Mtrx_Elist(H)[0]))
#    return H
