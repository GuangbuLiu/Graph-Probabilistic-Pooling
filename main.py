import argparse
import pdb
import time
import torch
import torch.nn.functional as F
from models import GraphClassificationModel
from utils import Logger
from utils import getModelSize
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from sklearn.model_selection import KFold,StratifiedShuffleSplit,StratifiedKFold
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--n_refpoint', type=int, default=32, help='number of dictionary')
parser.add_argument('--pool_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--kl_ratio', type=float, default=1, help='Ratio of KL-Divergence')
parser.add_argument('--ref_loss_ratio', type = float, default=5, help = 'Ratio of Dictionary loss')
parser.add_argument('--train_learning', type=float, default=True, help='whether perform train')

# sys.stdout = Logger('./result.log')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

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

    def train():
        test_loss_values = []

        t = time.time()
        model.train()

        best_test_acc=0
        for epoch in range(args.epochs):
            loss_train = 0.0
            correct = 0
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(args.device)
                out, kl_loss, ref_loss = model(data)
                loss = F.nll_loss(out, data.y) + args.kl_ratio * kl_loss + args.ref_loss_ratio * ref_loss
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
                  'ref_loss:{:.6f}'.format(ref_loss.item() * args.ref_loss_ratio),
                  'loss_train: {:.6f}'.format(loss_train),
                  'acc_train: {:.6f}'.format(acc_train), 'loss_test: {:.6f}'.format(loss_test),
                  'acc_test: {:.6f}'.format(acc_test), 'time: {:.6f}s'.format(time.time() - t))

            test_loss_values.append(loss_test)

        print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

        return best_test_acc


    def compute_test(loader):
        model.eval()
        correct = 0.0
        loss_test = 0.0
        for data in loader:
            data = data.to(args.device)
            out, kl_loss, ref_loss = model(data)
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


