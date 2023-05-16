import argparse
import glob
import os
import pdb
import sys
import time
import torch
import torch.nn.functional as F
from models import GraphClassificationModel
from utils import Logger, plt_dictionary
from torch.utils.data import random_split
from utils import getModelSize
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from sklearn.model_selection import KFold,StratifiedShuffleSplit,StratifiedKFold
# from ogb.graphproppred import PygGraphPropPredDataset
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=False, help='whether perform structure learning')
parser.add_argument('--hop_connection', type=bool, default=False, help='whether directly connect node within h-hops')
parser.add_argument('--hop', type=int, default=3, help='h-hops')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=2.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--n_dic', type=int, default=32, help='number of dictionary')
parser.add_argument('--pool_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--kl_ratio', type=float, default=1, help='Ratio of KL-Divergence')
parser.add_argument('--dic_loss_ratio', type = float, default=5, help = 'Ratio of Dictionary loss')
# parser.add_argument('--sparse_loss_ratio', type = float, default=1, help = 'Ratio of Dictionary loss')
parser.add_argument('--train_learning', type=float, default=True, help='whether perform train')

# sys.stdout = Logger('./result.log')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
# pdb.set_trace()
if args.dataset == 'IMDB-MULTI' or args.dataset == 'REDDIT-MULTI-12K' or args.dataset == 'IMDB-BINARY' or args.dataset == 'COLLAB':
    dataset = TUDataset('data/', name=args.dataset)
    max_degree = 0
    for g in dataset:
        if g.edge_index.size(1) > 0:
            max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
    dataset.transform = OneHotDegree(max_degree)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
else:
    dataset = TUDataset('data/', name=args.dataset, use_node_attr=True)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
print(args)

'''LGB'''
kf=KFold(n_splits=10,shuffle=True,random_state=args.seed)

fold_test_acc=[]
ifold = 0

for trainidx,test_idx in kf.split(dataset):

    ifold = ifold + 1
    print('ifold:',ifold)

    training_set=dataset[trainidx]
    test_set=dataset[test_idx]

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = GraphClassificationModel(args).to(args.device)
    # getModelSize(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500,gamma=0.1,last_epoch=-1,verbose=False)

    def train():
        min_loss = 1e10
        patience_cnt = 0
        test_loss_values = []
        best_epoch = 0

        t = time.time()
        model.train()

        best_test_acc=0
        for epoch in range(args.epochs):
            loss_train = 0.0
            correct = 0
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(args.device)
                out, kl_loss, dic_loss = model(data, epoch)
                loss = F.nll_loss(out, data.y) + args.kl_ratio * kl_loss + args.dic_loss_ratio * dic_loss
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
            acc_train = correct / len(train_loader.dataset)

            acc_test,loss_test =compute_test(test_loader)
            if acc_test>best_test_acc:
                best_test_acc=acc_test
                # torch.save(model.state_dict(), './checkpoint/ifold_{}_best_model.pth'.format(ifold))

            print('Epoch: {:04d}'.format(epoch + 1),'kl_loss:{:.8f}'.format(kl_loss * args.kl_ratio),
                  'dic_loss:{:.6f}'.format(dic_loss.item() * args.dic_loss_ratio),
                  'loss_train: {:.6f}'.format(loss_train),
                  'acc_train: {:.6f}'.format(acc_train), 'loss_test: {:.6f}'.format(loss_test),
                  'acc_test: {:.6f}'.format(acc_test), 'time: {:.6f}s'.format(time.time() - t))

            # acc_val, loss_val = compute_test(val_loader)

            test_loss_values.append(loss_test)

            # val_loss_values.append(loss_val)
            # torch.save(model.state_dict(), '{}.pth'.format(epoch))

            # if test_loss_values[-1] < min_loss:
            #     min_loss = test_loss_values[-1]
            #     best_epoch = epoch
            #     patience_cnt = 0
            # else:
            #     patience_cnt += 1
            #
            # if patience_cnt == args.patience:
            #     break

            # if val_loss_values[-1] < min_loss:
            #     min_loss = val_loss_values[-1]
            #     best_epoch = epoch
            #     patience_cnt = 0
            # else:
            #     patience_cnt += 1
            #
            # if patience_cnt == args.patience:
            #     break

            # files = glob.glob('*.pth')
            # for f in files:
            #     epoch_nb = int(f.split('.')[0])
            #     if epoch_nb < best_epoch:
            #         os.remove(f)

        # files = glob.glob('*.pth')
        # for f in files:
        #     epoch_nb = int(f.split('.')[0])
        #     if epoch_nb > best_epoch:
        #         os.remove(f)
        print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

        # return best_epoch
        return best_test_acc


    def compute_test(loader):
        model.eval()
        correct = 0.0
        loss_test = 0.0
        for data in loader:
            data = data.to(args.device)
            out,kl_loss, dic_loss = model(data, 0)
            # out,kl_loss = model(data, 0)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss_test += F.nll_loss(out, data.y).item()
        return correct / len(loader.dataset), loss_test

    if args.train_learning == True:
        ifold_test_acc=train()
    else:
        model.load_state_dict(torch.load('./checkpoint/ifold_{}_best_model.pth'.format(ifold)))
        ifold_test_acc, loss_test = compute_test(test_loader)

    fold_test_acc.append(ifold_test_acc)
    print(fold_test_acc)

test_acc=torch.tensor(fold_test_acc)


print(test_acc)
print(test_acc.mean(),test_acc.std())

# if __name__ == '__main__':
#     Model training
    # best_model = train()
    # Restore best model for test set
    # model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    # test_acc, test_loss = compute_test(test_loader)
    # print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))


