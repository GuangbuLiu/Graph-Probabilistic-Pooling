import pdb

import torch
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('The number of model Parameters:', param_sum)
    print('The size of model Parameters:', param_size)
    print('The total size of model Parameters:{:.3f}MB'.format(all_size))
    # return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    return all_size

def plt_dictionary(dictionary):
    # input:(N, d)
    dic = dictionary.clone().detach().cpu().numpy()
    tsne = TSNE(n_components = 2, random_state = 33)
    res = tsne.fit_transform(dic)
    plt.figure()
    plt.scatter(res[:, 0], res[0:, 1])
    plt.savefig('./fig/dictionary_distribution.png')

class Logger(object):
    def __init__(self,filename = './result.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # * the rest for testing
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data
