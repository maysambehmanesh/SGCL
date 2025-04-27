import torch
import numpy as np
import torch.nn.functional as F


from GCL.losses import Loss

def gaussian(x, sigma):
    return torch.exp(-0.5 * (x / sigma) ** 2)

    
def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()



def Taubin_smoothing(L, num, pos_mask, neg_mask, param):
    device = pos_mask.device
    pos_mask = pos_mask.to('cpu')
    val = torch.diag(torch.ones(num))
    I = torch.sparse_coo_tensor(val.nonzero().t(), val[val != 0], size=(num, num))     
    pos_mask0 = pos_mask  
    K = param.K
    ta =  param.tau
    mu = param.mu
    for i in range(0,K):
        pos_mask = torch.matmul((I + ta*L),pos_mask)
        pos_mask = torch.matmul((I + mu*L),pos_mask)
        pos_mask = (pos_mask0+(1-pos_mask0)*pos_mask)
        pos_mask[pos_mask<0.02]=0
        
    pos_mask = pos_mask.to(device)

    neg_mask = 1 - pos_mask

    return pos_mask, neg_mask


def bilateral_smoothing(edge_index, pos_mask, param):
    sigma_s = param.sigma_s
    sigma_r = param.sigma_r
    n = pos_mask.size(0)
    values_smooth = torch.zeros_like(pos_mask)

    
    for i in range(n):
        values = pos_mask[:,i]
        neighbors = edge_index[:, edge_index[0] == i]
        # spatial_weights = gaussian(neighbors[1] - i, sigma_s)
        spatial_weights = gaussian(torch.tensor(1), sigma_s)
        range_weights = gaussian(torch.abs(values[neighbors[1]] - values[i]), sigma_r)
        weights = spatial_weights * range_weights
        values_smooth[i,i] = 1
        values_smooth[neighbors[1],i] = weights
        
    pos_mask = pos_mask+(1-pos_mask)*values_smooth
    neg_mask = 1-pos_mask
    
    return pos_mask, neg_mask



def diffusion_based_smoothing(pos_mask, edge_index, param):
    n, m = pos_mask.shape
    value = pos_mask.clone()
    values_smooth = value
    
    num_iterations= param.num_iterations
    diffusion_rate= param.diffusion_rate
    
    for iteration in range(num_iterations):
        for i in range(n):
            neighbors = edge_index[1,:][edge_index[0]==i]
            diffusion_sum = torch.sum(values_smooth[neighbors,:],dim=0)
            values_smooth[i,:] = values_smooth[i,:] + diffusion_rate * diffusion_sum
            
    pos_mask = pos_mask+(1-pos_mask)*values_smooth
    neg_mask = 1-pos_mask

    return pos_mask, neg_mask
   
    
class HardnessJSD(Loss):
    def __init__(self, discriminator=lambda x, y: x @ y.t(), tau_plus=0.1, beta=0.05):
        super(HardnessJSD, self).__init__()
        self.discriminator = discriminator
        self.tau_plus = tau_plus
        self.beta = beta

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        num_pos = pos_mask.int().sum()
        similarity = self.discriminator(anchor, sample)

        pos_sim = similarity * pos_mask
        E_pos = np.log(2) - F.softplus(- pos_sim)
        E_pos -= (self.tau_plus / (1 - self.tau_plus)) * (F.softplus(-pos_sim) + pos_sim)
        E_pos = E_pos.sum() / num_pos

        neg_sim = similarity * neg_mask
        E_neg = F.softplus(- neg_sim) + neg_sim

        reweight = -2 * neg_sim / max(neg_sim.max(), neg_sim.min().abs())
        reweight = (self.beta * reweight).exp()
        reweight /= reweight.mean(dim=1, keepdim=True)

        E_neg = (reweight * E_neg) / (1 - self.tau_plus) - np.log(2)
        E_neg = E_neg.sum() / num_neg

        return E_neg - E_pos




class JSD(Loss):
    def __init__(self, discriminator=lambda x, y: x @ y.t()):
        super(JSD, self).__init__()
        self.discriminator = discriminator

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        num_pos = pos_mask.int().sum()
        similarity = self.discriminator(anchor, sample)

        E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
        E_pos /= num_pos

        neg_sim = similarity * neg_mask
        E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)).sum()
        E_neg /= num_neg

        return E_neg - E_pos
    
    

class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()
    
    
def c_similarity(A1, A2):
    A1 = torch.nn.functional.normalize(A1, p=2, dim=1)
    A2 = torch.nn.functional.normalize(A2, p=2, dim=1)
    S = torch.matmul(A1, A2.transpose(0, 1))
    S = 0.5 * (S + 1)
    
    return S
   
    
def bt_loss(h1: torch.Tensor, h2: torch.Tensor, pos_mask, L, edge_index, s_type, s_param, lambda_, batch_norm=True, eps=1e-15, *args, **kwargs):
    batch_size = h1.size(0)
    feature_dim = h1.size(1)
    neg_mask = 1-pos_mask
    
    if s_type=='Taubin':
        pos_mask, neg_mask = Taubin_smoothing(L, batch_size, pos_mask, 1-pos_mask, s_param)
    elif s_type=='Bilateral':
        pos_mask, neg_mask = bilateral_smoothing(edge_index, pos_mask, s_param)
    elif s_type == 'Diffusion_based':                        
        pos_mask, neg_mask = diffusion_based_smoothing(pos_mask, edge_index, s_param)
    else:
        print('Invalid smoothing approach')
    
    if lambda_ is None:
        lambda_ = 1. / batch_size

    if batch_norm:
        h1_n = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
        h2_n = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
        c = (h1_n @ h2_n.T) / feature_dim
    else:
        c = h1 @ h2.T / feature_dim
          
    # c = c_similarity(h1,h2)
    loss = ((1 - c)*pos_mask).pow(2).sum()
    loss += lambda_ *(c*neg_mask).pow(2).sum()
    
    return loss


class SGCL_loss(Loss):
    def __init__(self, lambda_: float = None, batch_norm: bool = True, eps: float = 1e-5):
        self.lambda_ = lambda_
        self.batch_norm = batch_norm
        self.eps = eps
        

    def compute(self, anchor, sample, pos_mask, neg_mask, L, edge_index, s_type, s_param, *args, **kwargs) -> torch.FloatTensor:
        loss = bt_loss(anchor, sample, pos_mask, L, edge_index, s_type, s_param, self.lambda_, self.batch_norm, self.eps)
        return loss.mean()
    
    
    
class BootstrapLatent(Loss):
    def __init__(self):
        super(BootstrapLatent, self).__init__()

    def compute(self, anchor, sample, pos_mask, neg_mask,L, edge_index, *args, **kwargs) -> torch.FloatTensor:
        ## Taubin smoothing
        batch_size = anchor.shape[0]
        pos_mask, neg_mask = Taubin_smoothing(L, batch_size, pos_mask, 1-pos_mask)

        
        anchor = F.normalize(anchor, dim=-1, p=2)
        sample = F.normalize(sample, dim=-1, p=2)

        similarity = anchor @ sample.t()
        loss = (similarity * pos_mask).sum(dim=-1)
        return loss.mean()
    
    
    



