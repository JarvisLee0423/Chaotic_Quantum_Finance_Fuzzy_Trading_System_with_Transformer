'''
    Copyright:      JarvisLee
    Date:           10/23/2021
    File Name:      ChaoticDecoder.py
    Description:    This file is used to generator the chaotic decoder.
'''

# Import the necessary library.
import torch
import torch.nn as nn
from Model.Modules import PositionalEmbedding
from Model.Layers import DecodeLayer

# Create the class for the chaotic decoder.
class ChaoticDecoder(nn.Module):
    '''
        The chaotic decoder.\n
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
        super(ChaoticDecoder, self).__init__()
        # Create the decoder layer.
        self.decoder = nn.ModuleList([
            DecodeLayer(head, dModel, dim, hidden, dropout, Lee, Mish) for _ in range(blocks)
        ])
        # Create the layer norm.
        self.layerNorm = nn.LayerNorm(dModel, eps = 1e-6)
    
    # Create the forward.
    def forward(self, query, key, value, posEmbed):
        # Compute the layer norm.
        query = self.layerNorm(query)
        key = self.layerNorm(key)
        value = self.layerNorm(value)
        # Compute the decoder.
        for layer in self.decoder:
            query = layer(query, key, value, posEmbed)
        # Return the result.
        return query

# Test the codes.
if __name__ == "__main__":
    query = torch.randn(32, 20, 100)
    key = torch.randn(32, 20, 100)
    value = torch.randn(32, 20, 100)
    PosEmbed = PositionalEmbedding()
    PE = PosEmbed(query)
    decoder = ChaoticDecoder(6, 8, 100, 300, 300, 0.1)
    output = decoder(query, key, value, PE)
    print(f'The output of the decoder (shape: {output.shape}):\n {output}')