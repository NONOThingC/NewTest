import torch.nn as nn
import torch
import torch.nn.functional as F
from .backbone import Bert_Encoder
import collections
import functools

class Encoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder = Bert_Encoder(args)
        self.output_size = self.encoder.out_dim
        dim_in = self.output_size
        self.head = nn.Sequential(nn.Linear(dim_in, dim_in),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(dim_in, args.feat_dim))
        self.grad_all = collections.defaultdict(list)

    def save_grad(self,grad, name):

        self.grad_all[name].append(grad.cpu())
        

    def reset_grad_recoder(self):
        self.grad_all = collections.defaultdict(list)

    def bert_forward(self, x, record_grad=False):
        out = self.encoder(x)
        if record_grad:
            out.register_hook(functools.partial(self.save_grad,name='first') )
        xx = self.head(out)
        xx = F.normalize(xx, p=2, dim=1)
        if record_grad:
            xx.register_hook(functools.partial(self.save_grad,name='second'))

        return out, xx
