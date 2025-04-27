import torch
import math
import numpy as np
import scipy.io as io
import os.path as osp
import torch_geometric.transforms as T
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid,Amazon,Coauthor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected, add_remaining_self_loops


def split_graph(data):
    return {
        'train': torch.nonzero(data.train_mask).squeeze().to('cpu'),
        'valid': torch.nonzero(data.val_mask).squeeze().to('cpu'),
        'test': torch.nonzero(data.test_mask).squeeze().to('cpu')}



def get_dataset(ds_name):
    """
    Get dataset
        Parameters:
            ds_name:  ame of dataset 

    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data')

    if ds_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, ds_name, transform=T.NormalizeFeatures())

    elif ds_name in ['Computers', 'Photo']:
        dataset = Amazon(path, ds_name, transform=T.NormalizeFeatures())

    elif ds_name == 'CoauthorCS':
        dataset = Coauthor(path, 'CS', transform=T.NormalizeFeatures())

    elif ds_name in ['ogbn-arxiv','ogbn-products']:
        dataset = PygNodePropPredDataset(
            name=ds_name, root=path, transform=T.ToSparseTensor())
    elif ds_name in ['ogbn-proteins']:     
        dataset = PygNodePropPredDataset(
            name=ds_name, root=path, transform=T.ToSparseTensor(attr='edge_attr'))
        dataset.data.x=dataset[0].adj_t.mean(dim=1)

    else:
        raise Exception('Unknown dataset.')

    
    return dataset




def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper

def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.max() + 1):
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=True))

  val_idx = [i for i in development_idx if i not in train_idx]

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  data.train_mask = get_mask(train_idx)
  data.val_mask = get_mask(val_idx)
  data.test_mask = get_mask(test_idx)

  return data




def split_data(dataset):
    # lcc = np.array(range(0,num_nodes))
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
    dataset.data = data

    dataset.data = set_train_val_test_split(
        12345,
        dataset.data,
        num_development=5000 if dataset.name == "CoauthorCS" else 1500)
    
    return dataset





def split_data_arxive(dataset):
    num_nodes = dataset.data.num_nodes
    split_idx = dataset.get_idx_split()
    ei = to_undirected(dataset.data.edge_index)
    data = Data(
        x=dataset.data.x,
        edge_index=ei,
        y=dataset.data.y,
        train_mask=split_idx['train'],
        test_mask=split_idx['test'],
        val_mask=split_idx['valid']
        )
    
    dataset.data = data
        
    train_mask= np.zeros(num_nodes)
    val_mask= np.zeros(num_nodes)
    test_mask= np.zeros(num_nodes)
        
    train_mask[data.train_mask]=1
    val_mask[data.val_mask]=1
    test_mask[data.test_mask]=1
    
    dataset.data.train_mask=torch.tensor(train_mask, dtype=bool)
    dataset.data.val_mask=torch.tensor(val_mask, dtype=bool)
    dataset.data.test_mask=torch.tensor(test_mask, dtype=bool)
    dataset.data.y=dataset.data.y.squeeze(-1)

    return dataset
    
  
   

def read_data(ds_name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

    dataset = split_data(Planetoid(path, ds_name, transform=T.NormalizeFeatures()))
    data=dataset[0]        
    # if ds_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS',]:
    #     dataset=split_data(get_dataset(ds_name))
    #     # dataset = get_dataset(ds_name)
    #     data=dataset[0]
    # if ds_name in ['ogbn-arxiv','ogbn-products']:
    #     dataset=split_data_arxive(get_dataset(ds_name))
    #     data=dataset.data
    # if ds_name in ['ogbn-proteins']:
    #     dataset=split_data_arxive(get_dataset(ds_name))
    #     data=dataset.data
    #     adj_t=dataset[0].adj_t
    #     data.y = torch.max(data.y,dim=1).indices

    return data, dataset




def get_split(num_samples: int, train_ratio: float = 0.1,  test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'test': indices[train_size: test_size + train_size],
        'valid': indices[test_size + train_size:]
    }
    
    
    
def get_split_main(data):
    return {
        'train': torch.nonzero(data.train_mask).squeeze(),
        'test': torch.nonzero(data.test_mask).squeeze(),
        'valid': torch.nonzero(data.val_mask).squeeze()
    }
 
