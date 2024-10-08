import pdb

import torch
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = sample_gumbel(logits.size()).to(device=logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)

    if hard:
        shape = y.size()
        _, max_idx = y.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, max_idx, 1.0)
        y = (y_hard - y).detach() + y

    return y
