# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from model.layers.quantumnn.embedding import PositionEmbedding
from model.layers.complexnn.multiply import ComplexMultiply
# from layers.quantumnn.mixture import QMixture
# from layers.quantumnn.rnn import QRNNCell
# from layers.quantumnn.measurement import QMeasurement
# from layers.complexnn.measurement import ComplexMeasurement
# from layers.quantumnn.outer import QOuter
# # from models.SimpleNet import SimpleNet
# from layers.complexnn.l2_norm import L2Norm
# from layers.quantumnn.dense import QDense
# from layers.quantumnn.dropout import QDropout


class QMN(nn.Module):
    def __init__(self, input_channel):
        super(QMN, self).__init__()
        self.input_dims = [input_channel, input_channel,input_channel, input_channel]
        self.total_input_dim = sum(self.input_dims)
        self.embed_dim = input_channel
        self.speaker_num = 2

        # MELD data
        # The one-hot vectors are not the global user ID
        self.projections = nn.ModuleList([nn.Conv1d(dim, self.embed_dim, 3, 1, 1) for dim in self.input_dims])

        self.multiply = ComplexMultiply()
        self.phase_embeddings = nn.ModuleList(
            [PositionEmbedding(input_channel, input_dim=4) for dim in self.input_dims])


    def forward(self, in_modalities):
        utterance_reps = [nn.ReLU()(projection(x)) for x, projection in zip(in_modalities, self.projections)]
        # Take the amplitudes
        amplitudes = [F.normalize(rep, dim=1) for rep in utterance_reps]
        
        phases = [phase_embed(smask) for smask, phase_embed in zip(in_modalities, self.phase_embeddings)]
        unimodal_pure = [self.multiply([phase, amplitude]) for phase, amplitude in zip(phases, amplitudes)]

        return unimodal_pure



# qmn = QMN(27)
# in_modalities = [torch.rand((8,27,512)), torch.rand(8,27,512), torch.rand(8,27,512), torch.rand(8,27,512)]
# print(qmn(in_modalities)[1][1].shape)