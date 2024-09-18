######################################################################
# Copyright 2021. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import scipy.sparse as sp
import numpy as np
#import tensorflow as tf
import torch
import torch.nn as nn
class PartialConnection(nn.Module):
    def __init__(self, adjlist, count,
                 activation=None,
                 use_bias=True,
                 device='cuda:0',
                 **kwargs):
        super(self.__class__, self).__init__()
        
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.units = len(adjlist)
        if activation is None:
            self.activation = None
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
        self.use_bias = use_bias
        self.device = device

        # Added for pcn
        self.adjlist = adjlist
        assert len(adjlist) >= 1

        self.input_dims_map = [8912, 17824, 35649, 71299]

        # From build()
        input_dim = self.input_dims_map[count] #x.shape[1]
        self.gen_adjmat_and_reshapemat(input_dim)

        self.kernel = nn.Parameter(torch.ones((1,self.num_edges_adjlist)).to(self.device), requires_grad=True)
        nn.init.xavier_uniform_(self.kernel)
        self.bias = nn.Parameter(torch.zeros((1,self.num_edges_adjlist)).to(self.device), requires_grad=True)
     

    def gen_adjmat_and_reshapemat(self, input_dim):
        # Generate adjmat
        count = 0
        ilist = []
        jlist = []
        for adj in self.adjlist:
            for j in adj:
                ilist += [count]
                jlist += [j]
                count +=1
        self.num_edges_adjlist = count
        adjmat = sp.coo_matrix((np.ones(len(ilist), dtype=np.float32), (ilist, jlist)), shape=(self.num_edges_adjlist, input_dim), dtype=np.float32)

        # Convert to sparse tensor
        values = adjmat.data
        v = torch.FloatTensor(values).to(self.device)
        indices = np.vstack((adjmat.row, adjmat.col))
        i = torch.LongTensor(indices).to(self.device)
        shape = adjmat.shape
        self.adjmat = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device)

        # Generate reshapemat
        count = 0
        ilist = []
        jlist = []
        for i, adj in enumerate(self.adjlist):
            for _ in adj:
                ilist += [i]
                jlist += [count]
                count+=1
        reshapemat = sp.coo_matrix((np.ones(len(ilist), dtype=np.float32), (ilist, jlist)), shape=(self.units, self.num_edges_adjlist), dtype=np.float32)
        
        # Convert to sparse tensor
        values = reshapemat.data
        v = torch.FloatTensor(values).to(self.device)
        indices = np.vstack((reshapemat.row, reshapemat.col))
        i = torch.LongTensor(indices).to(self.device)
        shape = reshapemat.shape
        self.reshapemat = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device)

        #print("PCN: Size {0} to size {1}".format(self.adjmat.shape[1], self.reshapemat.shape[0]))

    def forward(self, x):
        x_T = x.t()
        flat_f_T = torch.matmul(self.adjmat, x_T)
        flat_f = flat_f_T.t()
        # Weight + bias
        kernel_size = flat_f.shape[1]
        if self.use_bias:
            flat_f2 = flat_f * self.kernel + self.bias
        else:
            flat_f2 = flat_f * self.kernel

        flat_f2_T = flat_f2.t()
        output_T = torch.matmul(self.reshapemat, flat_f2_T)
        output = output_T.t()
        if self.activation is not None:
            output = self.activation(output)
        return output

