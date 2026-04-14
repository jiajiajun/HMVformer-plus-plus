# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class ComplexMultiply(torch.nn.Module):
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        phase = inputs[0]
        amplitude = inputs[1]
        
        if amplitude.dim() == phase.dim()+1: # Assigning each dimension with same phase
            cos = torch.unsqueeze(torch.cos(phase), dim=-1)
            sin = torch.unsqueeze(torch.sin(phase), dim=-1)
            
        elif amplitude.dim() == phase.dim(): #Each dimension has different phases
            cos = torch.cos(phase)
            sin = torch.sin(phase)
        
       
        else:
             raise ValueError('input dimensions of phase and amplitude do not agree to each other.')

        # print(cos.shape, amplitude.shape)

        real_part = cos * amplitude
        imag_part = sin * amplitude
        # print(real_part.shape)
        return [real_part, imag_part]