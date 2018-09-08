# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from collections import OrderedDict

from mxnet import gluon

from models.GraphNN.MPNN import MPNN


class GAT(MPNN):
    '''
    NOTE: This model is not yet generally useable - waiting for more sparse ops in mxnet

    Graph Attention Network from https://arxiv.org/pdf/1710.10903.pdf, but modified to have a different attention function per edge type
    (Averages the outputs of the multi-attention heads, and uses the DTNN readout function)
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


            self.readout_mlp = gluon.nn.HybridSequential()
            with self.readout_mlp.name_scope():
                self.readout_mlp.add(gluon.nn.Dense(self.hidden_size, activation='tanh', in_units=self.hidden_size))
                self.readout_mlp.add(gluon.nn.Dense(1, in_units=self.hidden_size))

    def compute_messages(self, F, hidden_states, edges, t):
        hidden_states = self.pre_attn_mapping(hidden_states)
        summed_msgs = []
        for key in self.attn_fxns.keys():
            adj_mat, heads = edges[key], self.attn_fxns[key]
            to_average = []
            for left_attn, right_attn, in heads:
                # Compute a^T[ Wh_i || Wh_j ] from the paper cited in the docstring by summing two sparse matrices,
                #    one containing a^T[ Wh_i ] and one containing a^T[ Wh_j ]
                left_pre_attn = left_attn(hidden_states) # n_vertices X 1
                right_pre_attn = right_attn(hidden_states) # n_vertices X 1
                # The right way to do this is staying with sparse matrices, but mxnet doesn't support all the ops yet
                # Broadcasting is done axis-aligned, so here we're broadcasting across rows with left_pre_attn
                pre_attns = left_pre_attn.T * adj_mat + adj_mat * right_pre_attn # n_vertices x n_vertices
                # Do the rest of the pre_attn operations on a^T[ Wh_i || Wh_j ]
                pre_attns = F.LeakyReLU(pre_attns, act_type='leaky', slope=0.2)
                pre_attns = F.exp(pre_attns)
                attns = pre_attns / F.sum(pre_attns, axis=1, keepdims=True)
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
