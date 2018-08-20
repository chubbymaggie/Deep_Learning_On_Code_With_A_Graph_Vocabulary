import itertools
import logging
import re
from collections import OrderedDict
from typing import List, Tuple

import mxnet as mx
import numpy as np
import scipy as sp
from mxnet import nd, gluon

from data.AugmentedAST import AugmentedAST
from data.Batch import Batch, NameGraphVocabInput
from experiments.utils import tuple_of_tuples_to_padded_array, PaddedArray
from models.VarNaming.CharCNN import VarNamingCharCNNDataPoint, VarNamingCharCNNDataEncoder, VarNamingCharCNN
from models.VarNaming.FixedVocab import VarNamingFixedVocabDataEncoder
from models.VarNaming.VarNamingModel import too_useful_edge_types

logger = logging.getLogger()


class VarNamingNameGraphVocabDataPoint(VarNamingCharCNNDataPoint):
    def __init__(self, *args, **kwargs):
        self.graph_vocab_node_real_names = None
        super().__init__(*args, **kwargs)


class VarNamingNameGraphVocabDataEncoder(VarNamingCharCNNDataEncoder):
    DataPoint = VarNamingNameGraphVocabDataPoint

    def __init__(self, *args, **kwargs):
        self.subtoken_flag = '__SUBTOKEN__'
        self.subtoken_edge_type = 'SUBTOKEN_USE'
        self.subtoken_reverse_edge_type = 'reverse_SUBTOKEN_USE'
        super().__init__(*args, **kwargs)
        self.all_node_types[self.subtoken_flag] = len(self.all_node_types)
        self.all_edge_types = frozenset(
            self.all_edge_types.union({self.subtoken_edge_type, self.subtoken_reverse_edge_type}))

    def encode(self, dp: VarNamingCharCNNDataPoint):
        sbtk_to_graph_vocab_node = {}
        dp.graph_vocab_node_real_names = []
        for i, data in dp.subgraph.nodes:
            if data['type'] == self.subtoken_flag:
                sbtk_to_graph_vocab_node[data['identifier']] = i
                dp.graph_vocab_node_real_names.append(data['identifier'])
        attn_label = [sbtk_to_graph_vocab_node.get(i, -1) for i in dp.label]
        super().encode(dp)
        dp.graph_vocab_node_real_names = tuple(dp.graph_vocab_node_real_names)
        # super's encoder gives the label in the vocabulary - here we add a label over the graph vocab nodes
        dp.label: Tuple[Tuple[int, ...], Tuple[int, ...]] = (dp.label, tuple(attn_label))


class VarNamingNameGraphVocab(VarNamingCharCNN):
    '''
    Model that embeds variable names into a Graph Vocabulary to do the VarNaming task
    '''

    DataEncoder = VarNamingNameGraphVocabDataEncoder
    InputClass = NameGraphVocabInput

    @staticmethod
    def extra_graph_processing(graph, instances, data_encoder):
        for node, data in list(graph.nodes):
            if graph.is_variable_node(node):
                node_subtokens = data_encoder.name_to_subtokens(data['identifier'])
                for st in node_subtokens:
                    st_node, _ = graph.add_node(st, identifier=st, type=data_encoder.subtoken_flag)
                    graph.add_edge(node, st_node, type=data_encoder.subtoken_edge_type)
                    graph.add_edge(st_node, node, type=data_encoder.subtoken_reverse_edge_type)
        return graph, instances

    @staticmethod
    def instance_to_datapoint(graph: AugmentedAST,
                              instance,
                              data_encoder: VarNamingFixedVocabDataEncoder,
                              max_nodes_per_graph: int = None):
        var_name, locs = instance

        name_me_flag = data_encoder.name_me_flag
        internal_node_flag = data_encoder.internal_node_flag

        subgraph = graph.get_containing_subgraph(locs, max_nodes_per_graph)

        # Flag the variables to be named
        for loc in locs:
            subgraph.nodes[loc]['identifier'] = name_me_flag
            edges_to_prune = subgraph.all_adjacent_edges(loc, too_useful_edge_types)
            subgraph._graph.remove_edges_from(edges_to_prune)

        # Remove any disconnected subtoken nodes (they could come from subtokens that are only in the name, and thus be unfair hints)
        for node, data in list(subgraph.nodes):
            if data['type'] == data_encoder.subtoken_flag and subgraph._graph.degree(node) == 0:
                subgraph._graph.remove_node(node)

        # Assemble node types, node names, and label
        subgraph.node_ids_to_ints_from_0()
        node_types = []
        node_names = []
        for node, data in sorted(subgraph.nodes):
            if subgraph.is_variable_node(node):
                node_types.append(sorted(list(set(re.split(r'[,.]', data['reference'])))))
                node_names.append(data['identifier'])
            else:
                node_types.append([data['type']])
                if data['type'] == data_encoder.subtoken_flag:
                    node_names.append(data['identifier'])
                else:
                    node_names.append(internal_node_flag)

        label = data_encoder.name_to_subtokens(var_name)

        return data_encoder.DataPoint(subgraph, node_types, node_names, var_name, label, graph.origin_file,
                                      data_encoder.encoder_hash)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.attn_decoder = gluon.nn.Dense(self.hidden_size, in_units=self.hidden_size, flatten=False)
            self.sentinel = self.params.get('sentinel', grad_req='write', shape=(1, self.hidden_size))

    def batchify(self, data_filepaths: List[str], ctx: mx.context.Context):
        '''
        Returns combined graphs and labels.
        Labels are a (PaddedArray, Tuple[str]) tuple.  The PaddedArray is size (batch x max_name_length) containing integers.
        The integer values correspond to the integers in this model's data encoder's all_node_name_subtokens dict,
           or if the integer value is greater than len(all_node_name_subtokens) it corresponds to which subtoken node
           in the graph represents the right subtoken
        '''
        data = [self.data_encoder.load_datapoint(i) for i in data_filepaths]

        # Get the size of each graph
        batch_sizes = nd.array([len(dp.node_names) for dp in data], dtype='int32', ctx=ctx)

        combined_node_types = tuple(itertools.chain(*[dp.node_types for dp in data]))
        subtoken_node_type_idx = self.data_encoder.all_node_types[self.data_encoder.subtoken_flag]
        graph_vocab_node_locations = [i for i in range(len(combined_node_types)) if
                                      combined_node_types[i][0] == subtoken_node_type_idx]
        graph_vocab_node_locations = np.array(graph_vocab_node_locations)
        graph_vocab_node_real_names = [dp.graph_vocab_node_real_names for dp in data]
        node_types = tuple_of_tuples_to_padded_array(combined_node_types, ctx)
        combined_node_names = tuple(itertools.chain(*[dp.node_names for dp in data]))
        target_locations = [i for i, name in enumerate(combined_node_names) if name == self.data_encoder.name_me_flag]
        node_names = []
        for name in combined_node_names:
            if name == self.data_encoder.internal_node_flag:
                node_names.append(self.data_encoder.name_to_1_hot('',
                                                                  embedding_size=self.data_encoder.max_name_encoding_length,
                                                                  mark_as_internal=True))
            elif name == self.data_encoder.name_me_flag:
                node_names.append(
                    self.data_encoder.name_to_1_hot('', embedding_size=self.data_encoder.max_name_encoding_length,
                                                    mark_as_special=True))
            else:
                node_names.append(
                    self.data_encoder.name_to_1_hot(name, embedding_size=self.data_encoder.max_name_encoding_length))
        node_names = nd.array(np.stack(node_names), dtype='float32', ctx=ctx)

        # Combine all the adjacency matrices into one big, disconnected graph
        edges = OrderedDict()
        for edge_type in self.data_encoder.all_edge_types:
            adj_mat = sp.sparse.block_diag([dp.edges[edge_type] for dp in data]).tocsr()
            adj_mat = nd.sparse.csr_matrix((adj_mat.data, adj_mat.indices, adj_mat.indptr), shape=adj_mat.shape,
                                           dtype='float32', ctx=ctx)
            edges[edge_type] = adj_mat

        # Get the real names of the variables we're supposed to be naming
        combined_fixed_vocab_labels = list(itertools.chain([dp.label[0] for dp in data]))
        # vocab labels are integers referring to indices in the model's data encoder's all_node_name_subtokens
        vocab_labels = tuple_of_tuples_to_padded_array(combined_fixed_vocab_labels, ctx,
                                                       pad_amount=self.max_name_length)
        combined_attn_labels = []
        for dp in data:
            graph_vocab_nodes_in_dp = [i for i in range(len(dp.node_types)) if
                                       dp.node_types[i][0] == subtoken_node_type_idx]
            combined_attn_labels.append(
                tuple([graph_vocab_nodes_in_dp.index(i) + 1 if i >= 0 else -1 for i in dp.label[1]]))
        # attn labels are integers referring to indices (+1 to avoid confusion with the padding value, which is 0) in the list of attn weights over graph vocab nodes the model will eventually output (or -1 if there's no appropriate node)
        attn_labels = tuple_of_tuples_to_padded_array(combined_attn_labels, ctx, pad_amount=self.max_name_length)
        attn_label = attn_labels.values
        subtoken_in_graph = attn_label > 0
        attn_label = len(
            self.data_encoder.all_node_name_subtokens) + attn_label - 1  # -1 because we're done avoiding the padding value
        # If the correct subtoken was in the graph, then pointing to it is the correct output (it will always be in the vocab during training)
        joint_label = PaddedArray(values=nd.where(subtoken_in_graph, attn_label, vocab_labels.values),
                                  value_lengths=vocab_labels.value_lengths)

        # Combine the (actual) real names of variables-to-be-named
        real_names = tuple([dp.real_variable_name for dp in data])

        data = self.InputClass(edges, node_types, node_names, batch_sizes, ctx, target_locations=target_locations,
                               graph_vocab_node_locations=graph_vocab_node_locations,
                               graph_vocab_node_real_names=graph_vocab_node_real_names)
        return Batch(data, [joint_label, real_names])

    def readout(self, F, hidden_states):
        '''
        Returns (batch x max_name_length x len(all_node_name_subtokens)) tensor of name predictions for each graph
        '''
        # Separate the batch back to having its own dimension, while grabbing the mean of the hidden states at the __NAME_ME!__ locations
        decoder_hid_states = []
        graph_vocab_hid_states = []
        graph_vocab_nodes_per_batch_element = []
        max_n_subtokens = 0
        length = 0
        for l in self.data.batch_sizes.asnumpy():
            locs_this_element = [loc for loc in self.data.target_locations if length <= loc < length + l]
            decoder_hid_states.append(F.mean(hidden_states[locs_this_element], axis=0, keepdims=True))
            graph_vocab_nodes_this_element = [loc for loc in self.data.graph_vocab_node_locations if
                                              length <= loc < length + l]
            graph_vocab_nodes_per_batch_element.append(len(graph_vocab_nodes_this_element))
            graph_vocab_hid_states.append(
                hidden_states[graph_vocab_nodes_this_element, :] if graph_vocab_nodes_this_element else None)
            max_n_subtokens = max(max_n_subtokens, len(graph_vocab_nodes_this_element))
            length += l
        decoder_hid_state = F.concat(*decoder_hid_states, dim=0)

        # Adding sentinel and padding graph_vocab_hid_states with 0s to the max length
        padded_hid_states = []
        for i, h in enumerate(graph_vocab_hid_states):
            if h is None:
                pad = F.zeros((max_n_subtokens, hidden_states.shape[1]), ctx=hidden_states.context)
                padded_hid_states.append(F.concat(self.sentinel.data(), pad, dim=0))
            else:
                pad_len = max_n_subtokens - h.shape[0]
                if pad_len:
                    pad = F.zeros((pad_len, h.shape[1]), ctx=h.context)
                    padded_hid_states.append(F.concat(self.sentinel.data(), h, pad, dim=0))
                else:
                    padded_hid_states.append(F.concat(self.sentinel.data(), h, dim=0))
        graph_vocab_hid_states = F.stack(*padded_hid_states, axis=0)
        # Expanding to sequence dimension (batch x words x seq x hid)
        graph_vocab_hid_states = graph_vocab_hid_states.expand_dims(axis=2).broadcast_axes(axis=2,
                                                                                           size=self.max_name_length)
        graph_vocab_nodes_per_batch_element = 1 + F.array(graph_vocab_nodes_per_batch_element, dtype='float32',
                                                          ctx=graph_vocab_hid_states.context)  # +1 for the sentinel

        # Get the hidden states from the decoder gru
        gru_outputs = []
        for _ in range(self.max_name_length):
            decoder_hid_state, _ = self.decoder_gru(
                F.zeros((decoder_hid_state.shape[0], 1), ctx=decoder_hid_state.context), [decoder_hid_state])
            gru_outputs.append(decoder_hid_state)
        gru_output = F.stack(*gru_outputs, axis=1)  # batch x seq x hid

        # Computing the probabilities for the attention and vocab outputs and stacking them together
        vocab_output = F.softmax(self.vocab_decoder(gru_output), axis=2)  # batch x seq x vocab_size
        attn_keys = self.attn_decoder(gru_output)
        attn_output = F.sum(attn_keys.expand_dims(1) * graph_vocab_hid_states, axis=3)
        attn_output = F.exp(attn_output)
        attn_output = F.SequenceMask(attn_output, use_sequence_length=True,
                                     sequence_length=graph_vocab_nodes_per_batch_element, axis=1)
        attn_probs = (attn_output / F.sum(attn_output, axis=1, keepdims=True)).swapaxes(1,
                                                                                        2)  # batch x seq x max_n_sbtk
        gating_value, attn_probs = attn_probs[:, :, :1], attn_probs[:, :, 1:]
        vocab_probs = gating_value * vocab_output
        joint_probs = F.concat(vocab_probs, attn_probs, dim=2)

        return joint_probs

    def unbatchify(self, batch, model_output):
        '''
        Returns predictions and labels, both as lists of strings
        '''
        _, real_names = batch.label
        predictions_labels = []
        output_preds = nd.argmax(model_output, axis=2).asnumpy().astype(int)
        for i, row in enumerate(output_preds):
            prediction = []
            for pred in row:
                if pred == self.data_encoder.all_node_name_subtokens['__PAD__']:
                    continue
                if pred < len(self.data_encoder.all_node_name_subtokens):
                    prediction.append(self.data_encoder.rev_all_node_name_subtokens[pred])
                else:
                    prediction.append(batch.data.graph_vocab_node_real_names[i][
                                          pred - len(self.data_encoder.all_node_name_subtokens)])
            predictions_labels.append((prediction, self.data_encoder.name_to_subtokens(real_names[i])))

        return predictions_labels
