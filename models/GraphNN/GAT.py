# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from collections import OrderedDict

import scipy as sp
from mxnet import gluon

from models.FITB.FITBModel import FITBModel
from models.GraphNN.MPNN import MPNN


class GAT(MPNN):
    '''
    Graph Attention Network from https://arxiv.org/pdf/1710.10903.pdf, but modified to have a different attention function per edge type
    (Averages the outputs of the multi-attention heads, uses relu instead of leakyrelu, and uses the DTNN readout function)
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs['hidden_size']
        self.n_multi_attention_heads = kwargs['n_multi_attention_heads']

        # Initializing model components
        with self.name_scope():
            self.pre_attn_mapping = gluon.nn.Dense(self.hidden_size, in_units=self.hidden_size, use_bias=False)
            self.attn_fxns = OrderedDict()
            for t in self.data_encoder.all_edge_types:
                heads = []
                for h in range(self.n_multi_attention_heads):
                    left_attn = gluon.nn.Dense(1, in_units=self.hidden_size)
                    self.register_child(left_attn)
                    right_attn = gluon.nn.Dense(1, in_units=self.hidden_size)
                    self.register_child(right_attn)
                    heads.append((left_attn, right_attn))
                self.attn_fxns[t] = heads

            if FITBModel in self.__class__.mro():
                self.readout_mlp = gluon.nn.HybridSequential()
                with self.readout_mlp.name_scope():
                    self.readout_mlp.add(gluon.nn.Dense(self.hidden_size, activation='tanh', in_units=self.hidden_size))
                    self.readout_mlp.add(gluon.nn.Dense(1, in_units=self.hidden_size))

    def compute_messages(self, F, hidden_states, edges, t):
        hidden_states = self.pre_attn_mapping(hidden_states)
        summed_msgs = []
        for key in self.attn_fxns.keys():
            adj_mat, heads = edges[key], self.attn_fxns[key]
            # Adding self-edges for numerical stability
            adj_mat = adj_mat + F.sparse.csr_matrix(sp.sparse.eye(*adj_mat.shape, dtype='float32', format='csr'), ctx=adj_mat.context)
            to_average = []
            for left_attn, right_attn, in heads:
                # Compute a^T[ Wh_i || Wh_j ] from the paper cited in the docstring by summing two sparse matrices,
                #    one containing a^T[ Wh_i ] and one containing a^T[ Wh_j ]
                left_pre_attn = left_attn(hidden_states) # n_vertices X 1
                right_pre_attn = right_attn(hidden_states) # n_vertices X 1
                # Broadcasting is done axis-aligned, so here we're broadcasting across rows with left_pre_attn
                pre_attns = adj_mat * left_pre_attn + adj_mat * right_pre_attn.T
                # Do the rest of the pre_attn operations on a^T[ Wh_i || Wh_j ]
                pre_attns = F.sparse.relu(pre_attns)
                pre_attns = F.sparse.expm1(pre_attns) + adj_mat # +adj_mat to compensate for the minus 1 in expm1
                attns = pre_attns / F.sparse.sum(pre_attns, axis=1, keepdims=True)
                # Compute attention-weighted sum of all neighbor representations (for this edge type)
                to_average.append(F.dot(attns, hidden_states))
            avg = F.mean(F.stack(*to_average), axis=0)
            summed_msgs.append(avg) # n_vertices X hidden_size
        summed_msgs = F.sum(F.stack(*summed_msgs), axis=0)
        return summed_msgs

    def update_hidden_states(self, F, hidden_states, messages, t):
        return F.LeakyReLU(messages, act_type='elu')

    def readout(self, F, hidden_states):
        return self.readout_mlp(hidden_states)
