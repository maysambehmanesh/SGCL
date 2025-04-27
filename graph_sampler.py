import numpy as np
from torch import nn
from tqdm import tqdm
from graph_saint import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTEdgeSampler2, EgoGraphSampler

def graph_smp(batch_type, data, batch_size, walk_length, num_steps, n_radious, sample_coverage, save_dir):
    if batch_type == 'RWS':
        loader = GraphSAINTRandomWalkSampler(data,batch_size=batch_size,
                                        walk_length=walk_length,
                                        num_steps=num_steps,
                                        sample_coverage=sample_coverage,
                                        save_dir=save_dir)    
    elif batch_type == 'Ego':
        loader = EgoGraphSampler(data, batch_size=batch_size,
                                        n_radious=n_radious,
                                        num_steps=num_steps,
                                        sample_coverage=sample_coverage,
                                        save_dir=save_dir)
    elif batch_type == 'NSP':
        loader = GraphSAINTNodeSampler(data, batch_size=batch_size,
                                        num_steps=num_steps,
                                        sample_coverage=sample_coverage,
                                        save_dir=save_dir)
    elif batch_type == 'ESP':
        loader = GraphSAINTEdgeSampler2(data,batch_size=batch_size,
                                        num_steps=num_steps,
                                        sample_coverage=sample_coverage,
                                        save_dir=save_dir)

    else:
        print(f'{batch_type} is an invalid batch sampler model')
    
    return loader
