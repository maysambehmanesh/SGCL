from typing import List, Union, Tuple, Optional
import numba
import numpy as np
import torch
from scipy import sparse
import scipy
import scipy.sparse as sp
from torch_geometric.utils import get_laplacian
from scipy.sparse import csr_matrix, coo_matrix
from torch_geometric.utils import coalesce
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    

def get_normalized_Adj(edge_index, edge_weight, num_nodes, n_type='with_sl'):

    if n_type=='with_sl':
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,fill_value=1., num_nodes=num_nodes)

    edge_weight = torch.ones(edge_index.size(1),device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = torch.pow(deg,-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    # edge_weight = 1- edge_weight
    L = (edge_index, edge_weight)
   
    return L



def compute_laplacian(edge_index, num_nodes,  normalization, device):
    edge_index = edge_index.to('cpu')

    if normalization in ['without_sl','with_sl']:
        L0 = get_normalized_Adj(edge_index, num_nodes=num_nodes, n_type=normalization)
    else:
        L0 = get_laplacian(edge_index, num_nodes=num_nodes, normalization=normalization)
    
    L = sparse.coo_matrix((L0[1].numpy(), (L0[0][0, :].numpy(), L0[0][1, :].numpy())), shape=(num_nodes, num_nodes))
    L = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(L)).to(device)
            
    return L