'''
    Copyright:      JarvisLee
    Date:           10/22/2021
    File Name:      Modules.py
    Description:    This file is used to set the necessary modules.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Model.LeeOscillator import Mish, LeeTanh, LeeReLu, LeeMish

# Create the class for the mask generate module.
class Mask():
    '''
        The module for generate the mask for the attention.\n
    '''
    # Create the function to generate the mask for the attention.
    def getMask(seq):
        # Get the sequence length.
        seqLen = seq.shape[1]
        # Create the mask.
        mask = (1 - torch.triu(torch.ones((1, 1, seqLen, seqLen), device = seq.device), diagonal = 1)).bool()
        # Return the mask. (mask.shape = [batchSize, head, seqLen, seqLen])
        return mask

# Create the class for the basic self-attention module.
class SelfAttention(nn.Module):
    '''
        The module for computing the self-attention.\n
        Params:\n
            - dk: The dimension of the key value.
            - dropout: The dropout of the self-attention.
    '''
    # Create the constructor.
    def __init__(self, dk, dropout = 0.1):
        # Create the super constructor.
        super(SelfAttention, self).__init__()
        # Get the member variables.
        self.dk = dk
        # Create the dropout layer.
        self.dropout = nn.Dropout(dropout)
    
    # Create the forward.
    def forward(self, q, k, v, mask = None):
        # Compute the self-sttention. (q.shape = k.shape = v.shape = [batchSize, head, seqLen, dim])
        selfAtten = torch.matmul(q, k.transpose(2, 3)) / (self.dk ** 0.5)
        # Check whether apply the mask.
        if mask is not None:
            selfAtten = selfAtten.masked_fill(mask == 0, -1e9)
        # Get the final attention. (selfAtten.shape = [batchSize, head, seqLen, seqLen])
        selfAtten = self.dropout(torch.softmax(selfAtten, dim = -1))
        # Return the output. (output.shape = [batchSize, head, seqLen, dim])
        return torch.matmul(selfAtten, v)

# Create the class for the multi-head self-attention module.
class MultiHeadAttention(nn.Module):
    '''
        The module for computing the multi-head self-attention.\n
        Params:\n
            - head: The number of the heads.
            - dModel: The original dimension of the input.
            - dim: The dimension of the query (q), key (k), and value (v).
            - dropout: The dropout of the multi-head self-attention.
    '''
    # Create the constructor.
    def __init__(self, head, dModel, dim, dropout = 0.1):
        # Create the super constructor.
        super(MultiHeadAttention, self).__init__()
        # Get the member variables.
        self.head = head
        self.dim = dim
        # Create the linear layer to convert the input data into q, k, and v.
        self.Wq = nn.Linear(dModel, head * dim, bias = False)
        self.Wk = nn.Linear(dModel, head * dim, bias = False)
        self.Wv = nn.Linear(dModel, head * dim, bias = False)
        # Create the linear layer to convert the self attention output into final output.
        self.Wo = nn.Linear(head * dim, dModel, bias = False)
        # Create the self-attention module.
        self.selfAtten = SelfAttention(dim, dropout)
        # Create the dropout layer.
        self.dropout = nn.Dropout(dropout)
        # Create the layer norm.
        self.layerNorm = nn.LayerNorm(dModel, eps = 1e-6)
    
    # Create the forward.
    def forward(self, query, key, value, mask = None):
        # Get the batch size and sequence length. (query.shape = key.shape = value.shape = [batchSize, seqLen, dModel])
        bs, seqLen = query.shape[0], query.shape[1]
        # Get the query, key, and value. (q.shape = k.shape = v.shape = [batchSize, head, seqLen, dim])
        q = self.Wq(query).reshape((bs, self.head, seqLen, self.dim))
        k = self.Wk(key).reshape((bs, self.head, seqLen, self.dim))
        v = self.Wv(value).reshape((bs, self.head, seqLen, self.dim))
        # Get the residul.
        residual = query
        # Compute the self-attention.
        output = self.selfAtten(q, k, v, mask)
        output = output.transpose(2, 1).reshape(bs, seqLen, self.head * self.dim)
        # Combine each head's output.
        output = self.dropout(self.Wo(output))
        # Combine the output and residual.
        output = output + residual
        # Compute the layer norm.
        output = self.layerNorm(output)
        # Return the output. (output.shape = [batchSize, seqLen, dModel])
        return output

# Create the class for the feed forward module.
class FeedForward(nn.Module):
    '''
        The module for computing the feed forward network.\n
        Params:\n
            - dModel: The input size of the fully-connected layer.
            - hidden: The hidden size of the fully-connected layer.
            - dropout: The dropout of the feed forward network.
            - Lee: The Lee-Oscillator based activation function.
            - Mish: The Mish activation function. 
    '''
    # Create the constructor.
    def __init__(self, dModel, hidden, dropout = 0.1, Lee = None, Mish = None):
        # Create the super constructor.
        super(FeedForward, self).__init__()
        # Get the member variables.
        self.Lee = Lee
        self.Mish = Mish
        # Create the fully-connected layer for the feed forward network.
        self.fc1 = nn.Linear(dModel, hidden)
        self.fc2 = nn.Linear(hidden, dModel)
        # Create the layer norm.
        self.layerNorm = nn.LayerNorm(dModel, eps = 1e-6)
        # Create the dropout layer.
        self.dropout = nn.Dropout(dropout)
    
    # Create the forward.
    def forward(self, x):
        # Get the residual. (x.shape = [batchSize, seqLen, dModel])
        residual = x
        # Compute the feed forward network.
        if self.Lee is not None:
            output = self.fc2(self.Lee(self.fc1(x)))
        elif self.Mish is not None:
            output = self.fc2(self.Mish(self.fc1(x)))
        else:
            output = self.fc2(torch.relu(self.fc1(x)))
        # Compute the dropout.
        output = self.dropout(output)
        # Combine the output and residual.
        output = output + residual
        # Compute the layer norm.
        output = self.layerNorm(output)
        # Return the output. (output.shape = [batchSize, seqLen, dModel])
        return output

# Create the class for the positional embedding module.
class PositionalEmbedding(nn.Module):
    '''
        The module for computing the positional embedding.\n
    '''
    # Create the constructor.
    def __init__(self):
        # Create the super constructor.
        super(PositionalEmbedding, self).__init__()
    
    # Create the forward.
    def forward(self, x, plot = False):
        # Initialize the positional embedding.
        PE = []
        # Get the positional embedding.
        for pos in range(x.shape[1]):
            PE.append([pos / np.power(10000, (2 * (i // 2) / x.shape[2])) for i in range(x.shape[2])])
        # Convert the positional embedding to be the tensor.
        PE = torch.tensor(PE, dtype = torch.float32)
        # Compute the positional embedding.
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        # Plot the positional embedding.
        if plot:
            # Print the positional embedding.
            #print(f'The positional embedding (shape: {PE.shape}):\n {PE}')
            plt.matshow(PE.cpu().numpy())
            plt.colorbar()
            plt.title('Positional Embedding Sample', pad = 20)
            plt.xlabel('Dimension of the input data')
            plt.ylabel('Sequence of the input data')
            plt.savefig("./PositionalEmbeddingSample.jpg")
            plt.show()
        # Return the positional embedding. (PE.shape = [1, seqLen, dim])
        return PE.unsqueeze(0).to(x.device).detach()

# Create the class for the chaotic embedding module.
class ChaoticEmbedding(nn.Module):
    '''
        The module for computing the chaotic embedding.\n
        Params:\n
            - a: The hyperparameters for Lee-Oscillator.
            - K: The K coefficient of Lee-Oscillator.
            - N: The number of iterations of Lee-Oscillator.
    '''
    # Create the constructor.
    def __init__(self, Lee):
        # Create the super constructor.
        super(ChaoticEmbedding, self).__init__()
        # Create the chaotic convertor.
        self.chaos = Lee
    
    # Create the forward.
    def forward(self, x, extractor, plot = False, filename = None):
        # Chaos the extractor.
        extractor = self.chaos(extractor)
        # Plot the chaotic extractor.
        if plot and filename is not None:
            #print(f'The chaotic features extractor (shape: {extractor.shape}):\n {extractor}')
            #plt.matshow(extractor.cpu().detach().numpy())
            #plt.colorbar()
            #plt.title('Chaotic Features Extractor', pad = 20)
            #plt.xlabel('Y Dimension of the extractor')
            #plt.ylabel('X Dimension of the extractor')
            #plt.savefig(f"./{filename}.jpg")
            #plt.close()
            #plt.show()
            with open(f"./{filename}.txt", "w") as file:
                file.write(str(extractor.cpu().detach().tolist()))
        # Get the chaotic embedding.
        CE = torch.matmul(extractor, x)
        # Return the chaotic embedding. (CE.shape = [1, seqLen, dim])
        return CE.to(x.device)

# Test the codes.
if __name__ == "__main__":
    x = torch.randn((2, 20, 46))
    a = torch.randn((2, 8, 20, 20))
    print(f'The original attention (shape: {a.shape}):\n {a}')
    mask = Mask.getMask(x)
    print(f'The mask (shape: {mask.shape}):\n {mask}')
    a = a.masked_fill(mask == 0, -1e9)
    print(f'The masked attention (shape: {a.shape}):\n {a}')
    a = nn.functional.softmax(a, dim = -1)
    print(f'The softmax of the masked attention (shape: {a.shape}):\n {a}')

    x = torch.randn((5, 20, 46))
    q = torch.randn((5, 8, 20, 46))
    k = torch.randn((5, 8, 20, 46))
    v = torch.randn((5, 8, 20, 46))
    mask = Mask.getMask(x)
    selfAtten = SelfAttention(k.shape[3])
    output = selfAtten(q, k, v, mask)
    print(f'The output of the self-attention (shape: {output.shape}):\n {output}')

    x = torch.randn((5, 20, 46))
    fc = nn.Linear(46, 100)
    input = fc(x)
    print(f'The input of the multi-head self-attention (shape: {input.shape}):\n {input}')
    w = torch.empty((20, 20))
    w = nn.init.xavier_normal_(w)
    PosEmbed = PositionalEmbedding()
    PE = PosEmbed(input, plot = False)
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    Lee = LeeTanh(a, K, N)
    ChaosEmbed = ChaoticEmbedding(Lee)
    CE = ChaosEmbed(input, w, plot = False)
    input = input + PE + CE
    print(f'The revised input of the multi-head self-attention (shape: {input.shape}):\n {input}')
    mask = Mask.getMask(input)
    MultiHeadAtten = MultiHeadAttention(8, 100, 300)
    output = MultiHeadAtten(input, input, input, mask)
    print(f'The output of the multi-head self-attention (shape: {output.shape}):\n {output}')

    x = torch.randn((5, 20, 46))
    fc = nn.Linear(46, 100)
    input = fc(x)
    print(f'The input of the feed forward network (shape: {input.shape}):\n {input}')
    mish = Mish()
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    Lee1 = LeeTanh(a, K, N)
    Lee2 = LeeReLu(a, K, N)
    Lee3 = LeeMish(a, K, N)
    FFN1 = FeedForward(100, 300, Lee = Lee1)
    FFN2 = FeedForward(100, 300, Lee = Lee2)
    FFN3 = FeedForward(100, 300, Lee = Lee3)
    FFN4 = FeedForward(100, 300, Mish = mish)
    FFN5 = FeedForward(100, 300)
    output = FFN1(input)
    print(f'The input of the feed forward network with Tanh based Lee-Oscillator (shape: {output.shape}):\n {output}')
    output = FFN2(input)
    print(f'The input of the feed forward network with ReLu based Lee-Oscillator (shape: {output.shape}):\n {output}')
    output = FFN3(input)
    print(f'The input of the feed forward network with Mish based Lee-Oscillator (shape: {output.shape}):\n {output}')
    output = FFN4(input)
    print(f'The input of the feed forward network with Mish (shape: {output.shape}):\n {output}')
    output = FFN5(input)
    print(f'The input of the feed forward network with ReLu (shape: {output.shape}):\n {output}')

    x = torch.randn((5, 100, 500))
    print(f'The input data (shape: {x.shape}):\n {x}')
    PosEmbed = PositionalEmbedding()
    x = x + PosEmbed(x, plot = True)
    print(f'The input data with positional embedding (shape: {x.shape}):\n {x}')

    w = torch.empty((20, 20))
    w = nn.init.xavier_normal_(w)
    print(f'The features extractor (shape: {w.shape}):\n {w}')
    x = torch.randn((5, 20, 46))
    print(f'The input data (shape: {x.shape}):\n {x}')
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    Lee = LeeTanh(a, K, N)
    ChaosEmbed = ChaoticEmbedding(Lee)
    x = x + ChaosEmbed(x, w, plot = True, filename = 'ChaoticFeaturesExtractorSample')
    print(f'The input data with chaotic embedding (shape: {x.shape}):\n {x}')