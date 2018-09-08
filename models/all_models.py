# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from models.FITB.CharCNN import FITBCharCNN
from models.FITB.FixedVocab import FITBFixedVocab
from models.FITB.NameGraphVocab import FITBNameGraphVocab
from models.GraphNN.DTNN import DTNN
from models.GraphNN.GAT import GAT
from models.GraphNN.GGNN import GGNN
from models.GraphNN.RGCN import RGCN
from models.VarNaming.CharCNN import VarNamingCharCNN
from models.VarNaming.FixedVocab import VarNamingFixedVocab
from models.VarNaming.NameGraphVocab import VarNamingNameGraphVocab


class FITBFixedVocabGGNN(FITBFixedVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNGGNN(FITBCharCNN, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBNameGraphVocabGGNN(FITBNameGraphVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingFixedVocabGGNN(VarNamingFixedVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNGGNN(VarNamingCharCNN, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingNameGraphVocabGGNN(VarNamingNameGraphVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBFixedVocabDTNN(FITBFixedVocab, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNDTNN(FITBCharCNN, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBNameGraphVocabDTNN(FITBNameGraphVocab, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingFixedVocabDTNN(VarNamingFixedVocab, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNDTNN(VarNamingCharCNN, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingNameGraphVocabDTNN(VarNamingNameGraphVocab, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBFixedVocabRGCN(FITBFixedVocab, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNRGCN(FITBCharCNN, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBNameGraphVocabRGCN(FITBNameGraphVocab, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingFixedVocabRGCN(VarNamingFixedVocab, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNRGCN(VarNamingCharCNN, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingNameGraphVocabRGCN(VarNamingNameGraphVocab, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBFixedVocabGAT(FITBFixedVocab, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNGAT(FITBCharCNN, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBNameGraphVocabGAT(FITBNameGraphVocab, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingFixedVocabGAT(VarNamingFixedVocab, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNGAT(VarNamingCharCNN, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingNameGraphVocabGAT(VarNamingNameGraphVocab, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
