'''
    Copyright:      JarvisLee
    Date:           10/23/2021
    File Name:      ChaoticEncoder.py
    Description:    This file is used to generate the chaotic encoder.
'''

# Import the necessary library.
import torch
import torch.nn as nn
from Model.Modules import Mask
from Model.Layers import EncodeLayer

# Create the class for the chaotic encoder.
class ChaoticEncoder(nn.Module):
    '''
        The chaotic encoder.\n
        Params:\n
            - blocks: The number of the encoder and decoder blocks will be applied into transformer.
            - head: The number of heads for the multi-head self-attention.
            - dModel: The original dimension of the input.
            - dim: The dimension of the query (q), key (k), and value (v).
            - hidden: The dimension of the hidden in fully-connected layer.
            - dropout: The dropout of the multi-head self-attention.
            - Lee: The Lee-Oscillator based activation function.
            - Mish: The mish activation function.
    '''
    # Create the constructor.
    def __init__(self, blocks, head, dModel, dim, hidden, dropout = 0.1, Lee = None, Mish = None):
        # Create the super constructor.
        super(ChaoticEncoder, self).__init__()
        # Create the encoder layer.
        self.encoder = nn.ModuleList([
            EncodeLayer(head, dModel, dim, hidden, dropout, Lee, Mish) for _ in range(blocks)
        ])
        # Create the layer norm.
        self.layerNorm = nn.LayerNorm(dModel, eps = 1e-6)
    
    # Create the forward.
    def forward(self, x, mask = None):
        # Compute the layer norm.
        output = self.layerNorm(x)
        # Compute the encoder.
        for layer in self.encoder:
            output = layer(output, mask)
        # Return the result.
        return output

# Test the codes.
if __name__ == "__main__":
    x = torch.randn((32, 20, 100))
    encoder = ChaoticEncoder(6, 8, 100, 300, 300, 0.1)
    mask = Mask.getMask(x)
    output = encoder(x, mask)
    print(f'The output of the encdoer (shape: {output.shape}):\n {output}')