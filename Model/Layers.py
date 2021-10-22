'''
    Copyright:      JarvisLee
    Date:           10/23/2021
    File Name:      Layers.py
    Description:    This file is used to set the necessary layers.
'''

# Import the necessary library.
import torch
import torch.nn as nn
from Model.Modules import Mask, MultiHeadAttention, FeedForward, PositionalEmbedding, ChaoticEmbedding
from Model.LeeOscillator import LeeTanh

# Create the class for the encode layer.
class EncodeLayer(nn.Module):
    '''
        The module for computing the encode layer.\n
        Params:\n
            - head: The number of heads for the multi-head self-attention.
            - dModel: The original dimension of the input.
            - dim: The dimension of the query (q), key (k), and value (v).
            - hidden: The dimension of the hidden in fully-connected layer.
            - dropout: The dropout of the multi-head self-attention.
            - Lee: The Lee-Oscillator based activation function.
            - Mish: The mish activation function.
    '''
    # Create the constructor.
    def __init__(self, head, dModel, dim, hidden, dropout = 0.1, Lee = None, Mish = None):
        # Craete the super constructor.
        super(EncodeLayer, self).__init__()
        # Create the multi-head self-attention.
        self.MHAtten = MultiHeadAttention(head, dModel, dim, dropout)
        # Create the feed forward layer.
        self.FFN = FeedForward(dModel, hidden, dropout, Lee, Mish)
    
    # Create the forward.
    def forward(self, x, mask = None):
        # Compute the multi-head self-attention.
        output = self.MHAtten(x, x, x, mask)
        # Compute the feed forward layer.
        output = self.FFN(output)
        # Return the result.
        return output
    
# Create the class for the decode layer.
class DecodeLayer(nn.Module):
    '''
        The module for computing the decode layer.\n
        Params:\n
            - head: The number of heads for the multi-head self-attention.
            - dModel: The original dimension of the input.
            - dim: The dimension of the query (q), key (k), and value (v).
            - hidden: The dimension of the hidden in fully-connected layer.
            - dropout: The dropout of the multi-head self-attention.
            - Lee: The Lee-Oscillator based activation function.
            - Mish: The mish activation function.
    '''
    # Create the constructor.
    def __init__(self, head, dModel, dim, hidden, dropout = 0.1, Lee = None, Mish = None):
        # Create the super constructor.
        super(DecodeLayer, self).__init__()
        # Create the first multi-head self-attention.
        self.MHAtten1 = MultiHeadAttention(head, dModel, dim, dropout)
        # Create the second multi-head self-attention.
        self.MHAtten2 = MultiHeadAttention(head, dModel, dim, dropout)
        # Create the feed forward layer.
        self.FFN = FeedForward(dModel, hidden, dropout, Lee, Mish)
    
    # Create the forward.
    def forward(self, query, key, value, posEmbed, chaoticEmbed = None):
        # Compute the first multi-head self-attention.
        output = self.MHAtten1(query + posEmbed, query + posEmbed, query + posEmbed)
        # Compute the second multi-head self-attention.
        if chaoticEmbed is not None:
            output = self.MHAtten2(output, key + posEmbed, value + posEmbed + chaoticEmbed)
        else:
            output = self.MHAtten2(output, key + posEmbed, value + posEmbed)
        # Compute the feed forward layer.
        output = self.FFN(output)
        # Return the result.
        return output

# Test the code.
if __name__ == "__main__":
    x = torch.randn((32, 20, 46))
    fc = nn.Linear(46, 100)
    x = fc(x)
    PosEmbed = PositionalEmbedding()
    PE = PosEmbed(x, plot = True)
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    Lee1 = LeeTanh(a, K, N)
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    K = 50
    N = 100
    Lee2 = LeeTanh(a, K, N)
    a = [0.55, 0.55, -0.5, 0.5, -0.55, -0.55, 0.5, -0.5]
    K = 50
    N = 100
    Lee3 = LeeTanh(a, K, N)
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    K = 300
    N = 100
    Lee4 = LeeTanh(a, K, N)
    a = [-0.2, 0.45, 0.6, 1, 0, -0.55, 0.55, 0]
    K = 100
    N = 100
    Lee5 = LeeTanh(a, K, N)
    Chaos1 = ChaoticEmbedding(Lee1)
    Chaos2 = ChaoticEmbedding(Lee2)
    Chaos3 = ChaoticEmbedding(Lee3)
    Chaos4 = ChaoticEmbedding(Lee4)
    Chaos5 = ChaoticEmbedding(Lee5)
    w1 = torch.empty((20, 20))
    w1 = nn.Parameter(nn.init.xavier_normal_(w1))
    CE1 = Chaos1(x, w1)
    w2 = torch.empty((20, 20))
    w2 = nn.Parameter(nn.init.xavier_normal_(w2))
    CE2 = Chaos2(x, w2)
    w3 = torch.empty((20, 20))
    w3 = nn.Parameter(nn.init.xavier_normal_(w3))
    CE3 = Chaos3(x, w3)
    w4 = torch.empty((20, 20))
    w4 = nn.Parameter(nn.init.xavier_normal_(w4))
    CE4 = Chaos4(x, w4)
    w5 = torch.empty((20, 20))
    w5 = nn.Parameter(nn.init.xavier_normal_(w5))
    CE5 = Chaos1(x, w5)
    CE = CE1 + CE2 + CE3 + CE4 + CE5
    x = x + PE + CE
    mask = Mask.getMask(x)
    query = nn.Embedding(20, 100)

    encoder = EncodeLayer(8, 100, 300, 300, 0.1, Lee1)
    encode = encoder(x, mask)
    decoder = DecodeLayer(8, 100, 300, 300, 0.1, Lee1)
    decode = decoder(query.weight.unsqueeze(0).repeat(32, 1, 1), encode, encode, PE, CE)
    print(f'The encode of the encoder (shape: {encode.shape}):\n {encode}')
    print(f'The decode of the decoder (shape: {decode.shape}):\n {decode}')