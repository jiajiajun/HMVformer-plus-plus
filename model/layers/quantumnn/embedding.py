# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import numpy as np

def PhaseEmbedding(input_dim, embedding_dim, freeze = False):
    phase_embedding_matrix = torch.empty(input_dim,embedding_dim)
    nn.init.uniform_(phase_embedding_matrix,0, 2*np.pi)
    embedding_layer = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=freeze)
    return embedding_layer

def FrequencyEmbedding(input_dim, embedding_dim, freeze = False):
    phase_embedding_matrix = torch.empty(input_dim,embedding_dim)
    nn.init.uniform_(phase_embedding_matrix,0, 10)
    embedding_layer = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=freeze)
    return embedding_layer

class PositionEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, input_dim = 1, zero_phase = False, device = torch.device('cpu')):
        super(PositionEmbedding, self).__init__()
        # self.device = device
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.zero_phase = zero_phase
        
        #Vaswani et al.
        # frequency_inits = 1 / torch.pow(10000, torch.true_divide(torch.arange(embed_dim//2), embed_dim//2))
        # frequency_matrix = frequency_inits.repeat(self.embed_dim, 1)
        # self.frequency_embedding = nn.Conv2d(embed_dim, self.embed_dim, 3, 1, 1)
        # self.positions = torch.arange(H * W)
        # phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        # self.phase_embedding = nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1)
        phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        self.phase_embedding = nn.Conv1d(embed_dim, self.embed_dim, 3, 1, 1)

        # position_matrix = torch.rand(self.input_dim, H * W)
        self.position_embeeding = nn.Conv2d(embed_dim, self.embed_dim, 3, 1, 1)

        
        #self.frequencies = nn.Parameter(frequency_inits.unsqueeze(dim = 0).to(self.device))
    
        
    def forward(self, x):
            
        # No speaker embedding
        if self.input_dim == 1:
            x = torch.zeros_like(x)
        phases = self.phase_embedding(x)
        phases = 2 * 3.14 * nn.Sigmoid()(phases)
        
        return phases

