import argparse
import torch
import GCL.augmentors as A
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from util_lap import compute_laplacian
from logistic_regression import LREvaluator
from TestEvaluator import LREvaluator_transfer
from GCL_models import WithinEmbedContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from graph_sampler import graph_smp
from load_data import read_data, get_split, get_split_main
from loss_function import SGCL_loss


parser = argparse.ArgumentParser(description='Smooth GCL (Cora)')

parser.add_argument('--source_dataset', type=str, default='Cora') 
parser.add_argument('--target_dataset', type=str, default='Cora')
parser.add_argument('--s_type',  type=str, default='Taubin') 
parser.add_argument('--encoder', type=str, default='GCN') 
parser.add_argument('--sampler', type=str, default='Ego') 
parser.add_argument('--hidden_channels', type=int, default=512) 
parser.add_argument('--num_layers', type=int, default=2) 
parser.add_argument('--lr', type=float, default=0.001) 
parser.add_argument('--epochs', type=int, default=100) 
parser.add_argument('--normalization', type=str, default='sym') 
parser.add_argument('--tau', type=float, default= 0.3, help='Taubin param') 
parser.add_argument('--mu', type=float, default= -0.4, help='Taubin param')
parser.add_argument('--K', type=int, default= 2, help='Taubin param')
parser.add_argument('--sigma_s', type=float, default= 0.1, help='Bilateral param') 
parser.add_argument('--sigma_r', type=float, default= 2, help='Bilateral param')
parser.add_argument('--num_iterations', type=int, default= 2, help='Diffusion_based param') 
parser.add_argument('--diffusion_rate', type=float, default= 0.03, help='Diffusion_based param')
parser.add_argument('--er', type=float, help="Edge removing", default=0.3)
parser.add_argument('--fm', type=float, help="Feature masking", default=0.3)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=2000) # 2000
parser.add_argument('--num_steps', type=int, default=4)# 4
parser.add_argument('--walk_length', type=int, default=3)   
parser.add_argument('--n_radious', type=int, default=1)   
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--load_CL', type=int, default=0)
parser.add_argument('--rate', type=float, default=0, help='Perturbation Probability')
args = parser.parse_args()

transfer = False if args.source_dataset == args.target_dataset else True

type_normalization = args.normalization
if args.normalization is None:
    type_normalization = 'unnormalized'
        


print(args)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)



data, dataset = read_data(args.source_dataset)
target_data, target_dataset = read_data(args.target_dataset)

    
in_features_src = dataset.num_features
in_features_trg = target_dataset.num_features


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
                
    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z



class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)

        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        
        return z1, z2, x1, x2, edge_index1, edge_index2
    
    def evaluate(self, x, edge_index, edge_weight=None):
        z = self.encoder1(x, edge_index, edge_weight)
        return z



def train(epoch, encoder_model, contrast_model, loader, optimizer, args, device):
    
    if epoch > args.load_CL:
        total_loss=0
        loss = 0
        for data in loader:
            data = data.to(device)
            encoder_model.train()
            optimizer.zero_grad()
          
            z1, z2, x1, x2, edge_index1, edge_index2 = encoder_model(data.x, data.edge_index)
            
            if args.s_type=='Taubin': 
                L1 = compute_laplacian(edge_index=edge_index1, num_nodes=x1.shape[0], normalization=args.normalization, device='cpu')
                L2 = compute_laplacian(edge_index=edge_index2, num_nodes=x2.shape[0], normalization=args.normalization, device='cpu')
            else:
                L1=L2=0

            loss = loss + contrast_model(h1=z1, h2=z2, L1=L1, L2=L2, edge_index1=data.edge_index, edge_index2=edge_index2, s_type=args.s_type, s_param=args)

            
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss




def test_transfer(encoder_model, in_features_trg, in_features_src, hidden_features, data, device):
    data = data.to(device)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    if args.target_dataset in ['Cora', 'Citeseer', 'Pubmed']:
        split = get_split_main(data)
    else:
        split = get_split(num_samples=data.x.size()[0], train_ratio=0.1, test_ratio=0.8)
        
    target_eval = LREvaluator_transfer(train_encoder= encoder_model, in_features_trg = in_features_trg, in_features_src = in_features_src, hidden_features=hidden_features)
    result = target_eval.evaluate(data.x, data.edge_index, data.y, split)

    return result

    


def test(encoder_model, data, device):
    data = data.to(device)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    z = encoder_model.evaluate(data.x, data.edge_index)
    if args.target_dataset in ['Cora', 'Citeseer', 'Pubmed']:
        split = get_split_main(data)
    else:
        split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)

    result = LREvaluator()(z, data.y, split)

    return result


trn_loader = graph_smp(batch_type=args.sampler, data=data, batch_size=args.batch_size, walk_length=args.walk_length,
                   num_steps=args.num_steps, n_radious=args.n_radious, sample_coverage=0, save_dir=dataset.processed_dir)

num_features = dataset.num_features

aug1 = A.Compose([A.EdgeRemoving(pe=args.er), A.FeatureMasking(pf=args.fm)])
aug2 = A.Compose([A.EdgeRemoving(pe=args.er), A.FeatureMasking(pf=args.fm)])
gconv1 = GConv(input_dim=num_features, hidden_dim=args.hidden_channels, num_layers=args.num_layers).to(device)
gconv2 = GConv(input_dim=num_features, hidden_dim=args.hidden_channels, num_layers=args.num_layers).to(device)
encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=args.hidden_channels).to(device)

contrast_model = WithinEmbedContrast(loss=SGCL_loss(),mode='L2L').to(device)   
optimizer = Adam(encoder_model.parameters(), lr=args.lr)


print(f'Pre-training on {args.source_dataset}')
with tqdm(total=args.epochs, desc='(T)') as pbar:
    for epoch in range(1, args.epochs+1):
        loss = train(epoch, encoder_model, contrast_model, trn_loader, optimizer, args, device)       
        pbar.set_postfix({'loss': loss})
        pbar.update()


print(f'Fine-tuning on {args.source_dataset}')
micro_f1, macro_f1, accuracy, roc_auc =[],[],[],[]
if transfer: 
    for i in range(0,2):
        test_result = test_transfer(encoder_model, in_features_trg, in_features_src, args.hidden_channels, target_data, device)
        micro_f1.append(test_result["micro_f1"])
        macro_f1.append(test_result["macro_f1"])
        accuracy.append(test_result["accuracy"])
        roc_auc.append(test_result["ROC-AUC"])
    print(f'(E): Best test F1Mi={np.mean(micro_f1)*100:2.2f}\xB1{np.std(micro_f1)*100:.2}, F1Ma={np.mean(macro_f1)*100:2.2f}\xB1{np.std(macro_f1)*100:.2}, ROC-AUC={np.mean(roc_auc)*100:2.2f}\xB1{np.std(roc_auc)*100:.2}, Acc={np.mean(accuracy)*100:2.2f}\xB1{np.std(accuracy)*100:.2}')

else:
    for i in range(0,2):
        test_result = test(encoder_model, data, device='cpu')
        micro_f1.append(test_result["micro_f1"])
        macro_f1.append(test_result["macro_f1"])
        accuracy.append(test_result["accuracy"])
        roc_auc.append(test_result["ROC-AUC"])
    print(f'(E): Best test F1Mi={np.mean(micro_f1)*100:2.2f}\xB1{np.std(micro_f1)*100:.2}, F1Ma={np.mean(macro_f1)*100:2.2f}\xB1{np.std(macro_f1)*100:.2}, ROC-AUC={np.mean(roc_auc)*100:2.2f}\xB1{np.std(roc_auc)*100:.2}, Acc={np.mean(accuracy)*100:2.2f}\xB1{np.std(accuracy)*100:.2}')

