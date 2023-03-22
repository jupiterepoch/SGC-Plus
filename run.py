import argparse
import string
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import CorrectAndSmooth
import torch_sparse

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.linkproppred import Evaluator as LinkEvaluator
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator as NodeEvaluator
from encoder import GCN, TAGC, SGC, GCNWithAttention, SGCwithJK, SAGEwithJK,  GCN, SGCWithAttention, Linear
from decoder import LinkPredictor, DotPredictor, MLPCatPredictor, MLPDotPredictor, MLPBilPredictor, BilinearPredictor, NodePredictor
from logger import Logger
from tqdm import tqdm
import wandb

def train(model, predictor, data, split_edge, optimizer, batch_size, margin=None):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device) 

    total_loss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):

        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        pos_edge = pos_train_edge[perm].t()
        neg_edge = neg_train_edge[perm].t()
        pos_out = predictor(h[pos_edge[0]], h[pos_edge[1]])
        neg_out = predictor(h[neg_edge[0]], h[neg_edge[1]])

        # random element of previously sampled negative edges
        # negative samples are obtained by using spatial sampling criteria

        if not margin:
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            loss = pos_loss + neg_loss
        else:
            margin = nn.Parameter(torch.Tensor([margin]), requires_grad=False).to(h.device)
            # print(pos_out.max(), pos_out.min(), pos_out.mean())
            # print(neg_out.max(), neg_out.min(), neg_out.mean())
            # print((neg_out - pos_out).mean())
            # print(pos_out.shape, neg_out.shape)
            loss = torch.max(pos_out - neg_out, -margin).mean() + margin

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_rocauc = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_train_pred,
        })[f'rocauc']

    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
        })[f'rocauc']

    test_rocauc = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

def train_node(model, predictor, data, split_node, optimizer, batch_size):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    for perm in DataLoader(split_node['train'], batch_size, shuffle=True):
        
        h = model(data.x, data.adj_t)

        optimizer.zero_grad()
        h = model(data.x, data.adj_t)

        logits = predictor(h[perm])
        labels = data.y[perm].squeeze(-1)

        # loss = F.cross_entropy(logits, labels, reduction='sum')
        loss = F.nll_loss(logits, labels, reduction='sum')
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(split_node['train'])

@torch.no_grad()
def test_node(model, predictor, data, split_node, evaluator, batch_size, smooth=False):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    deg = data.adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * data.adj_t * deg_inv_sqrt.view(1, -1)
    DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * data.adj_t

    post = CorrectAndSmooth(num_correction_layers=50, correction_alpha=0.9,
                            num_smoothing_layers=50, smoothing_alpha=0.8,
                            autoscale=True, scale=20.)

    train_idx = split_node['train']
    y_soft = predictor(h).softmax(-1)

    if smooth:
        print('Correct and smooth...')
        y_true = data.y[train_idx]
        y_soft = post.correct(y_soft, y_true, train_idx, DAD)
        y_soft = post.smooth(y_soft, y_true, train_idx, DAD).argmax(-1, keepdims=True)
    
    else:
        y_soft = y_soft.argmax(-1, keepdims=True)
    
    # pos_train_preds = []
    # for perm in DataLoader(split_node['train'], batch_size):
    #     pos_train_preds += [predictor(h[perm]).argmax(-1).squeeze(-1)]
    # pos_train_pred = torch.cat(pos_train_preds, dim=0).unsqueeze(-1)
    pos_train_pred = y_soft[train_idx]
    pos_train_true = data.y[train_idx]

    # pos_valid_preds = []
    # for perm in DataLoader(split_node['valid'], batch_size):
    #     pos_valid_preds += [predictor(h[perm]).argmax(-1).squeeze(-1)]
    # pos_valid_pred = torch.cat(pos_valid_preds, dim=0).unsqueeze(-1)
    pos_valid_pred = y_soft[split_node['valid']]
    pos_valid_true = data.y[split_node['valid']]

    # pos_test_preds = []
    # for perm in DataLoader(split_node['test'], batch_size):
    #     pos_test_preds += [predictor(h[perm]).argmax(-1).squeeze(-1)]
    # pos_test_pred = torch.cat(pos_test_preds, dim=0).unsqueeze(-1)
    pos_test_pred = y_soft[split_node['test']]
    pos_test_true = data.y[split_node['test']]

    train_acc = evaluator.eval({
            'y_true': pos_train_true,
            'y_pred': pos_train_pred,
        })[f'acc']

    valid_acc = evaluator.eval({
            'y_true': pos_valid_true,
            'y_pred': pos_valid_pred,
        })[f'acc']

    test_acc = evaluator.eval({
            'y_true': pos_test_true,
            'y_pred': pos_test_pred,
        })[f'acc']

    return train_acc, valid_acc, test_acc

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed_everything(seed)

def add_global_node(adj_t):

    adj_t = adj_t.to_torch_sparse_csr_tensor()
    num_nodes = adj_t.size(0)

    new_crow_indices = adj_t.crow_indices()
    new_crow_indices = torch.cat([new_crow_indices, torch.tensor([new_crow_indices[-1]+num_nodes+1])], dim=0)
    new_col_indices = adj_t.col_indices()
    new_col_indices = torch.cat([new_col_indices, torch.arange(num_nodes+1)], dim=0)
    new_values = adj_t.values()
    new_row = torch.ones(num_nodes+1, dtype=torch.float)
    new_row[-1] = 0
    new_values = torch.cat([new_values, new_row], dim=0)

    new_adj_t = torch.sparse_csr_tensor(new_crow_indices, new_col_indices, new_values, dtype=torch.float)

    new_adj_t = torch_sparse.tensor.SparseTensor.from_torch_sparse_csr_tensor(new_adj_t)
    # print(new_adj_t.to_symmetric().to_dense())

    return new_adj_t

def aug_normalized_adjacency(adj_t):
    adj_t = adj_t.to_torch_sparse_csr_tensor()
    num_nodes = adj_t.size(0)

    prev = 0
    new_crow_indices = adj_t.crow_indices()
    new_col_indices = adj_t.col_indices()
    new_values = adj_t.values()
    diags = []
    # print(new_crow_indices)
    for crow in new_crow_indices[1:]:
        num_idx = crow - prev
        row = new_values[prev:prev+num_idx]
        row_sum = row.sum().item()
        if row_sum == 0: row_sum = 1
        diags.append(row_sum ** -0.5)
        prev = crow
    diags = torch.tensor(diags)
    d_inv_sqrt = torch.sparse_csr_tensor(torch.arange(num_nodes+1), torch.arange(num_nodes), diags)
    normalized = torch.sparse.mm(torch.sparse.mm(d_inv_sqrt, adj_t), d_inv_sqrt)
    normalized = torch_sparse.tensor.SparseTensor.from_torch_sparse_csr_tensor(normalized)
    return normalized

def main():
    parser = argparse.ArgumentParser(description='OGBL-VESSEL (GNN) Algorithm.')
    
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--encoder_name', type=str, default='gcn', help='sage, lrga, gcn, sgc, sgcwjk, sagewjk, tagc')
    parser.add_argument('--decoder_name', type=str, default='mlp', help='mlp, dot, mlpcat, mlpdot, mlpbil, bil')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--mlp_layers', type=int, default=1)
    parser.add_argument('--mlp_dropout', type=float, default=0.4)
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--data_name', type=str, default='ogbl-vessel')
    parser.add_argument('--res_dir', type=str, default='log/')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=1e-6) 
    parser.add_argument('--epochs', type=int, default=10000) 
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--task', type=str, default='link')
    parser.add_argument('--margin', type=float, default=None)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--add_global_node', action='store_true')
    parser.add_argument('--smooth', action='store_true')

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.task == 'link':
        dataset = PygLinkPropPredDataset('ogbl-vessel', transform=T.ToSparseTensor())
    elif args.task == 'node':
        dataset = PygNodePropPredDataset('ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]
     
    # # normalize x,y,z coordinates  
    # data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
    # data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
    # data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)
    data.x = data.x.to(torch.float)
    if args.add_global_node:
        data.adj_t = add_global_node(data.adj_t)
        data.x = torch.vstack([data.x, torch.rand(1, data.num_features)])
    data.adj_t = data.adj_t.to_symmetric()
    # data.adj_t = aug_normalized_adjacency(data.adj_t)

    if args.use_node_embedding:
        data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)
    data = data.to(device)

    if args.task == 'link':
        split_edge = dataset.get_edge_split()
    elif args.task == 'node':
        split_node = dataset.get_idx_split()

    # create log file and save args
    log_file_name = 'log_' + args.data_name + '_' + str(int(time.time())) + '.txt'
    log_file = os.path.join(args.res_dir, log_file_name)
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')

    if args.encoder_name.lower() == 'sage':
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    # elif args.encoder_name.lower() == 'lrga':
    #     model = GCNWithAttention(data.num_features, args.hidden_channels,
    #                  args.hidden_channels, args.num_layers,
    #                  args.dropout).to(device)
    # elif args.encoder_name.lower() == 'gcn':
    #     model = GCN(data.num_features, args.hidden_channels,
    #                 args.hidden_channels, args.num_layers,
    #                 args.dropout).to(device)
    elif args.encoder_name.lower() == 'sgc':
        model = SGC(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, args.K).to(device)
    elif args.encoder_name.lower() == 'linear':
        model = Linear(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout, args.K).to(device)
    elif args.encoder_name.lower() == 'sgcwjk':
        model = SGCwithJK(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder_name.lower() == 'sagewjk':
        model = SAGEwithJK(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder_name.lower() == 'tagc':
        model = TAGC(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder_name.lower() == 'sgclrga':
        model = SGCWithAttention(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    else:
        print('Wrong model name!')

        # # Pre-compute GCN normalization.
        # adj_t = data.adj_t.set_diag()
        # deg = adj_t.sum(dim=1).to(torch.float)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        # data.adj_t = adj_t
    if args.decoder_name.lower() == 'mlp':
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                args.mlp_layers, args.mlp_dropout).to(device)
    # elif args.decoder_name.lower() == 'dot':
    #     predictor = DotPredictor(args.hidden_channels, args.hidden_channels, 1,
    #                             args.mlp_layers, args.mlp_dropout).to(device)
    # elif  args.decoder_name.lower() == 'mlpcat':
    #     predictor = MLPCatPredictor(args.hidden_channels, args.hidden_channels, 1,
    #                             args.mlp_layers, args.mlp_dropout).to(device)
    # elif  args.decoder_name.lower() == 'mlpdot':
    #     predictor = MLPDotPredictor(args.hidden_channels, args.hidden_channels, 1,
    #                             args.mlp_layers, args.mlp_dropout).to(device)
    # elif  args.decoder_name.lower() == 'mlpbil':
    #     predictor = MLPBilPredictor(args.hidden_channels, args.hidden_channels, 1,
    #                             args.mlp_layers, args.mlp_dropout).to(device)
    # elif  args.decoder_name.lower() == 'bil':
    #     predictor = BilinearPredictor(args.hidden_channels, args.hidden_channels, 1,
    #                             args.mlp_layers, args.mlp_dropout).to(device)
    elif  args.decoder_name.lower() == 'node':
        predictor = NodePredictor(args.hidden_channels, args.hidden_channels, dataset.num_classes,
                                args.mlp_layers, args.mlp_dropout).to(device)
    else:
        print('Wrong predictor name!')

    if args.task == 'link':
        evaluator = LinkEvaluator(name='ogbl-vessel')
    elif args.task == 'node':
        evaluator = NodeEvaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)   

    # 
    sum_params = 0.
    for p in model.parameters():
        sum_params += p.numel()
    for p in predictor.parameters():
        sum_params += p.numel()
    print(f'Params: {sum_params}')

    ''' start wandb '''
    if args.wandb:
        wandb.init(project="sgc", entity="cs224wfinal", name=args.name, config=args)
        wandb.watch(model)
        wandb.watch(predictor)
    ''''''

    train_roc_auc_list=[]
    valid_roc_auc_list=[]
    for run in range(args.runs):
        set_seed(run)
                
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.AdamW( # AdamW = Adam > SGD
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        ''' test before train '''
        # result = test(model, predictor, data, split_edge, evaluator, args.batch_size)
        # logger.add_result(run, result)
        # train_roc_auc, valid_roc_auc, test_roc_auc = result
        # result_dic = {
        #     'Epoch': 0,
        #     'Train': train_roc_auc,
        #     'Valid': valid_roc_auc,
        #     'Test': test_roc_auc
        # }
        # if args.wandb: wandb.log(result_dic)
        # print(f'Run: {run + 1:02d}, ', end=' ')
        # for key, val in result_dic.items():
        #     print(f'{key}: {val:.7f}, ', end=' ')
        # print('\n'+'*'*50)
        ''' end test '''

        for epoch in range(1, 1 + args.epochs):
            if args.task == 'link':
                loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, args.margin)
            elif args.task == 'node':
                loss = train_node(model, predictor, data, split_node, optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                if args.task == 'link':
                    result = test(model, predictor, data, split_edge, evaluator, args.batch_size)
                elif args.task == 'node':
                    result = test_node(model, predictor, data, split_node, evaluator, args.batch_size, args.smooth)
                logger.add_result(run, result)

                train_roc_auc, valid_roc_auc, test_roc_auc = result
                result_dic = {
                    'Epoch': epoch,
                    'Train': train_roc_auc,
                    'Valid': valid_roc_auc,
                    'Test': test_roc_auc,
                    "Loss": loss
                }
                if args.wandb: wandb.log(result_dic)
                print(f'Run: {run + 1:02d}, ', end=' ')
                for key, val in result_dic.items():
                    print(f'{key}: {val:.7f}, ', end=' ')
                print()
                train_roc_auc_list.append(train_roc_auc)
                valid_roc_auc_list.append(valid_roc_auc)
            # if epoch>3 and (valid_roc_auc_list[epoch-1]- valid_roc_auc_list[epoch-2])<1e-6 and valid_roc_auc_list[epoch-1]- valid_roc_auc_list[epoch-3]<1e-6:
            #     print('reset parameters')
            #     model.reset_parameters()
            #     predictor.reset_parameters()
        print('GNN')
        logger.print_statistics(run)

    print('GNN')
    logger.print_statistics()

if __name__ == "__main__":
    main()
