'''
    Copyright:      JarvisLee
    Date:           10/23/2021
    File Name:      ChaoticTransformer.py
    Description:    This file is used to generate the chaotic transformer.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Model.LeeOscillator import Mish, LeeTanh, LeeMish, LeeReLu
from Model.Modules import Mask, PositionalEmbedding, ChaoticEmbedding
from Model.ChaoticEncoder import ChaoticEncoder
from Model.ChaoticDecoder import ChaoticDecoder

# Create the class for the chaotic transformer.
class ChaoticTransformer(nn.Module):
    '''
        The chaotic transformer.\n
        Params:\n
            - blocks: The number of the encoder and decoder blocks will be applied into transformer.
            - head: The number of heads for the multi-head self-attention.
            - seqLen: The sequence length of the input data.
            - input: The original dimension of the input data.
            - output: The original dimension of the output data.
            - dModel: The original dimension of the input embedding.
            - dim: The dimension of the query (q), key (k), and value (v).
            - hidden: The dimension of the hidden in fully-connected layer.
            - dropout: The dropout of the multi-head self-attention.
            - decoder: The boolean to indicate whether apply the decoder or not.
            - mish: The boolean to indicate whether apply the mish activation function.
            - leeTanh: The boolean to indicate whether apply the Tanh based Lee-Oscillator.
            - leeReLu: The boolean to indicate whether apply the ReLu based Lee-Oscillator.
            - leeMish: The boolean to indicate whether apply the Mish based Lee-Oscillator.
            - chaotic: The type of the chaotic activation function.
            - chaosExtractor: The type of the chaotic extractor ('all' for using all five Lee-Oscillator).
    '''
    # Create the constructor.
    def __init__(self, blocks, head, seqLen, input, output, dModel, dim, hidden, dropout, decoder = True, mish = True, chaotic = False, leeTanh = True, leeReLu = False, leeMish = False, chaosExtractor = 'all'):
        # Create the super constructor.
        super(ChaoticTransformer, self).__init__()
        # Create the dictionary for the type of the Lee-Oscillator.
        LeeType = {'A':[[0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5], 50, 100], 
                    'B':[[1, 1, 1, 1, -1, -1, -1, -1], 50, 100], 
                    'C':[[0.55, 0.55, -0.5, 0.5, -0.55, -0.55, 0.5, -0.5], 50, 100], 
                    'D':[[1, 1, 1, 1, -1, -1, -1, -1], 300, 100], 
                    'E':[[-0.2, 0.45, 0.6, 1, 0, -0.55, 0.55, 0], 100, 100]}
        # Create the positional embedding.
        self.posEmbed = PositionalEmbedding()
        # Check the type of the basic function of the Lee-Oscillator.
        if leeTanh:
            # Check whether apply the chaos extractor.
            if chaosExtractor == 'all':
                # Print the hint.
                print(f'Contain the chaotic feature extractor with type {chaosExtractor} and with the Tanh based Lee-Oscillator.')
                # Get the Lee-Oscillator.
                Lee1 = LeeTanh(LeeType['A'][0], LeeType['A'][1], LeeType['A'][2])
                Lee2 = LeeTanh(LeeType['B'][0], LeeType['B'][1], LeeType['B'][2])
                Lee3 = LeeTanh(LeeType['C'][0], LeeType['C'][1], LeeType['C'][2])
                Lee4 = LeeTanh(LeeType['D'][0], LeeType['D'][1], LeeType['D'][2])
                Lee5 = LeeTanh(LeeType['E'][0], LeeType['E'][1], LeeType['E'][2])
                # Get the chaotic embedding generator.
                self.CE1 = ChaoticEmbedding(Lee1)
                self.CE2 = ChaoticEmbedding(Lee2)
                self.CE3 = ChaoticEmbedding(Lee3)
                self.CE4 = ChaoticEmbedding(Lee4)
                self.CE5 = ChaoticEmbedding(Lee5)
                self.extractor1 = torch.empty((seqLen, seqLen))
                self.extractor1 = nn.Parameter(nn.init.xavier_normal_(self.extractor1))
                self.extractor2 = torch.empty((seqLen, seqLen))
                self.extractor2 = nn.Parameter(nn.init.xavier_normal_(self.extractor2))
                self.extractor3 = torch.empty((seqLen, seqLen))
                self.extractor3 = nn.Parameter(nn.init.xavier_normal_(self.extractor3))
                self.extractor4 = torch.empty((seqLen, seqLen))
                self.extractor4 = nn.Parameter(nn.init.xavier_normal_(self.extractor4))
                self.extractor5 = torch.empty((seqLen, seqLen))
                self.extractor5 = nn.Parameter(nn.init.xavier_normal_(self.extractor5))
            elif chaosExtractor:
                # Print the hint.
                print(f'Contain the chaotic feature extractor with type {chaosExtractor} and with the Tanh based Lee-Oscillator.')
                # Get the Lee-Oscillator.
                Lee1 = LeeTanh(LeeType[chaosExtractor][0], LeeType[chaosExtractor][1], LeeType[chaosExtractor][2])
                # Get the chaotic embedding generator.
                self.CE1 = ChaoticEmbedding(Lee1)
                self.CE2 = None
                self.CE3 = None
                self.CE4 = None
                self.CE5 = None
                self.extractor1 = torch.empty((seqLen, seqLen))
                self.extractor1 = nn.Parameter(nn.init.xavier_normal_(self.extractor1))
            else:
                # Print the hint.
                print(f'Do not contain the chaotic feature extractor.')
                # Get the chaotic embedding generator.
                self.CE1 = None
                self.CE2 = None
                self.CE3 = None
                self.CE4 = None
                self.CE5 = None
            # Check whether apply the chaotic activation function.
            if chaotic:
                # Print the hint.
                print(f'Contain the type {chaotic} Tanh based Lee-Oscillator in the Feed Forward Layer.')
                # Get the chaotic activation function.
                Lee = LeeTanh(LeeType[chaotic][0], LeeType[chaotic][1], LeeType[chaotic][2])
                MishFunc = None
            else:
                Lee = None
                # Check whether apply the Mish.
                if mish:
                    # Print the hint.
                    print(f'Contain the Mish activation function in the Feed Forward Layer.')
                    MishFunc = Mish()
                else:
                    # Print the hint.
                    print(f'Contain the ReLu activation function in the Feed Forward Layer.')
                    MishFunc = None
        elif leeReLu:
            # Check whether apply the chaos extractor.
            if chaosExtractor == 'all':
                # Print the hint.
                print(f'Contain the chaotic feature extractor with type {chaosExtractor} and with the ReLu based Lee-Oscillator.')
                # Get the Lee-Oscillator.
                Lee1 = LeeReLu(LeeType['A'][0], LeeType['A'][1], LeeType['A'][2])
                Lee2 = LeeReLu(LeeType['B'][0], LeeType['B'][1], LeeType['B'][2])
                Lee3 = LeeReLu(LeeType['C'][0], LeeType['C'][1], LeeType['C'][2])
                Lee4 = LeeReLu(LeeType['D'][0], LeeType['D'][1], LeeType['D'][2])
                Lee5 = LeeReLu(LeeType['E'][0], LeeType['E'][1], LeeType['E'][2])
                # Get the chaotic embedding generator.
                self.CE1 = ChaoticEmbedding(Lee1)
                self.CE2 = ChaoticEmbedding(Lee2)
                self.CE3 = ChaoticEmbedding(Lee3)
                self.CE4 = ChaoticEmbedding(Lee4)
                self.CE5 = ChaoticEmbedding(Lee5)
                self.extractor1 = torch.empty((seqLen, seqLen))
                self.extractor1 = nn.Parameter(nn.init.xavier_normal_(self.extractor1))
                self.extractor2 = torch.empty((seqLen, seqLen))
                self.extractor2 = nn.Parameter(nn.init.xavier_normal_(self.extractor2))
                self.extractor3 = torch.empty((seqLen, seqLen))
                self.extractor3 = nn.Parameter(nn.init.xavier_normal_(self.extractor3))
                self.extractor4 = torch.empty((seqLen, seqLen))
                self.extractor4 = nn.Parameter(nn.init.xavier_normal_(self.extractor4))
                self.extractor5 = torch.empty((seqLen, seqLen))
                self.extractor5 = nn.Parameter(nn.init.xavier_normal_(self.extractor5))
            elif chaosExtractor:
                # Print the hint.
                print(f'Contain the chaotic feature extractor with type {chaosExtractor} and with the ReLu based Lee-Oscillator.')
                # Get the Lee-Oscillator.
                Lee1 = LeeReLu(LeeType[chaosExtractor][0], LeeType[chaosExtractor][1], LeeType[chaosExtractor][2])
                # Get the chaotic embedding generator.
                self.CE1 = ChaoticEmbedding(Lee1)
                self.CE2 = None
                self.CE3 = None
                self.CE4 = None
                self.CE5 = None
                self.extractor1 = torch.empty((seqLen, seqLen))
                self.extractor1 = nn.Parameter(nn.init.xavier_normal_(self.extractor1))
            else:
                # Print the hint.
                print(f'Do not contain the chaotic feature extractor.')
                # Get the chaotic embedding generator.
                self.CE1 = None
                self.CE2 = None
                self.CE3 = None
                self.CE4 = None
                self.CE5 = None
            # Check whether apply the chaotic activation function.
            if chaotic:
                # Print the hint.
                print(f'Contain the type {chaotic} ReLu based Lee-Oscillator in the Feed Forward Layer.')
                # Get the chaotic activation function.
                Lee = LeeReLu(LeeType[chaotic][0], LeeType[chaotic][1], LeeType[chaotic][2])
                MishFunc = None
            else:
                Lee = None
                # Check whether apply the Mish.
                if mish:
                    # Print the hint.
                    print(f'Contain the Mish activation function in the Feed Forward Layer.')
                    MishFunc = Mish()
                else:
                    # Print the hint.
                    print(f'Contain the ReLu activation function in the Feed Forward Layer.')
                    MishFunc = None
        elif leeMish:
            # Check whether apply the chaos extractor.
            if chaosExtractor == 'all':
                # Print the hint.
                print(f'Contain the chaotic feature extractor with type {chaosExtractor} and with the Mish based Lee-Oscillator.')
                # Get the Lee-Oscillator.
                Lee1 = LeeMish(LeeType['A'][0], LeeType['A'][1], LeeType['A'][2])
                Lee2 = LeeMish(LeeType['B'][0], LeeType['B'][1], LeeType['B'][2])
                Lee3 = LeeMish(LeeType['C'][0], LeeType['C'][1], LeeType['C'][2])
                Lee4 = LeeMish(LeeType['D'][0], LeeType['D'][1], LeeType['D'][2])
                Lee5 = LeeMish(LeeType['E'][0], LeeType['E'][1], LeeType['E'][2])
                # Get the chaotic embedding generator.
                self.CE1 = ChaoticEmbedding(Lee1)
                self.CE2 = ChaoticEmbedding(Lee2)
                self.CE3 = ChaoticEmbedding(Lee3)
                self.CE4 = ChaoticEmbedding(Lee4)
                self.CE5 = ChaoticEmbedding(Lee5)
                self.extractor1 = torch.empty((seqLen, seqLen))
                self.extractor1 = nn.Parameter(nn.init.xavier_normal_(self.extractor1))
                self.extractor2 = torch.empty((seqLen, seqLen))
                self.extractor2 = nn.Parameter(nn.init.xavier_normal_(self.extractor2))
                self.extractor3 = torch.empty((seqLen, seqLen))
                self.extractor3 = nn.Parameter(nn.init.xavier_normal_(self.extractor3))
                self.extractor4 = torch.empty((seqLen, seqLen))
                self.extractor4 = nn.Parameter(nn.init.xavier_normal_(self.extractor4))
                self.extractor5 = torch.empty((seqLen, seqLen))
                self.extractor5 = nn.Parameter(nn.init.xavier_normal_(self.extractor5))
            elif chaosExtractor:
                # Print the hint.
                print(f'Contain the chaotic feature extractor with type {chaosExtractor} and with the Mish based Lee-Oscillator.')
                # Get the Lee-Oscillator.
                Lee1 = LeeMish(LeeType[chaosExtractor][0], LeeType[chaosExtractor][1], LeeType[chaosExtractor][2])
                # Get the chaotic embedding generator.
                self.CE1 = ChaoticEmbedding(Lee1)
                self.CE2 = None
                self.CE3 = None
                self.CE4 = None
                self.CE5 = None
                self.extractor1 = torch.empty((seqLen, seqLen))
                self.extractor1 = nn.Parameter(nn.init.xavier_normal_(self.extractor1))
            else:
                # Print the hint.
                print(f'Do not contain the chaotic feature extractor.')
                # Get the chaotic embedding generator.
                self.CE1 = None
                self.CE2 = None
                self.CE3 = None
                self.CE4 = None
                self.CE5 = None
            # Check whether apply the chaotic activation function.
            if chaotic:
                # Print the hint.
                print(f'Contain the type {chaotic} Mish based Lee-Oscillator in the Feed Forward Layer.')
                # Get the chaotic activation function.
                Lee = LeeMish(LeeType[chaotic][0], LeeType[chaotic][1], LeeType[chaotic][2])
                MishFunc = None
            else:
                Lee = None
                # Check whether apply the Mish.
                if mish:
                    # Print the hint.
                    print(f'Contain the Mish activation function in the Feed Forward Layer.')
                    MishFunc = Mish()
                else:
                    # Print the hint.
                    print(f'Contain the ReLu activation function in the Feed Forward Layer.')
                    MishFunc = None
        else:
            # Print the hint.
            print(f'Do not contain the chaotic feature extractor.')
            # Get the chaotic activation function.
            Lee = None
            # Check whether apply the Mish.
            if mish:
                # Print the hint.
                print(f'Contain the Mish activation function in the Feed Forward Layer.')
                MishFunc = Mish()
            else:
                # Print the hint.
                print(f'Contain the ReLu activation function in the Feed Forward Layer.')
                MishFunc = None
            # Get the chaotic embedding generator.
            self.CE1 = None
            self.CE2 = None
            self.CE3 = None
            self.CE4 = None
            self.CE5 = None
        # Create the mask layer.
        self.mask = Mask
        # Create the embedding layer.
        self.embed = nn.Linear(input, dModel)
        # Print the hint.
        print(f'Apply the encoder.')
        # Create the encoder.
        self.encoder = ChaoticEncoder(blocks, head, dModel, dim, hidden, dropout, Lee, MishFunc)
        # Create the decoder.
        if decoder:
            # Print the hint.
            print(f'Apply the decoder.')
            # Create the output query.
            self.query = torch.empty((seqLen, dModel))
            self.query = nn.Parameter(nn.init.xavier_normal_(self.query))
            self.decoder = ChaoticDecoder(blocks, head, dModel, dim, hidden, dropout, Lee, MishFunc)
        else:
            # Print the hint.
            print(f'Do not apply the decoder.')
            self.decoder = None
        # Create the predictor.
        self.predictor = nn.Linear(seqLen * dModel, output)

    # Create the forward.
    def forward(self, x, plot = False, filename = 'Sample'):
        # Get the batch size.
        bs = x.shape[0]
        # Compute the embedding.
        x = self.embed(x)
        # Compute the positional embedding.
        PE = self.posEmbed(x)
        # Compute the chaotic embedding.
        CE = []
        if self.CE1 is not None:
            CE.append(self.CE1(x, self.extractor1, plot = plot, filename = f'{filename}_CE1'))
        if self.CE2 is not None:
            CE.append(self.CE2(x, self.extractor2, plot = plot, filename = f'{filename}_CE2'))
        if self.CE3 is not None:
            CE.append(self.CE3(x, self.extractor3, plot = plot, filename = f'{filename}_CE3'))
        if self.CE4 is not None:
            CE.append(self.CE4(x, self.extractor4, plot = plot, filename = f'{filename}_CE4'))
        if self.CE5 is not None:
            CE.append(self.CE5(x, self.extractor5, plot = plot, filename = f'{filename}_CE5'))
        # Combine the x with the chaotic embedding.
        if CE == []:
            CE = None
        else:
            CE = sum(CE)
            x = x + CE
        # Compute the mask.
        mask = self.mask.getMask(x)
        # Compute the input data.
        x = x + PE
        # Compute the encoder.
        encoding = self.encoder(x, mask)
        # Draw the query out.
        if self.query is not None and filename != 'Sample':
            #print(f'The query (shape: {query.shape}):\n {query}')
            #plt.matshow(self.query.cpu().detach().numpy())
            #plt.colorbar()
            #plt.title('Output Query', pad = 20)
            #plt.xlabel('Features Dimension')
            #plt.ylabel('Sequence Length')
            filenameList = filename.split('//')
            filenameList[2] = "OutputQuery"
            #plt.savefig(f"{'//'.join(filenameList)}.jpg")
            #plt.close()
            #plt.show()
            with open(f"{'//'.join(filenameList)}.txt", "w") as file:
                file.write(str(self.query.cpu().detach().numpy()))
        # Compute the result.
        if self.decoder is not None:
            # Compute the decoder.
            output = self.decoder(self.query.unsqueeze(0).repeat(bs, 1, 1), encoding, encoding, PE, CE)
            # Compute the output.
            return self.predictor(output.reshape(bs, -1)), mask.squeeze().cpu().detach().numpy()
        else:
            # Compute the output.
            return self.predictor(encoding.reshape(bs, -1)), mask.squeeze().cpu().detach().numpy()

# Test the code.
if __name__ == "__main__":
    blocks = 6 
    head = 8
    seqLen = 20
    input = 46
    output = 4
    dModel = 100 
    dim = 300 
    hidden = 300 
    dropout = 0.1 
    decoder = True
    mish = True
    chaotic = False 
    leeTanh = True
    leeReLu = False 
    leeMish = False 
    chaosExtractor = 'all'
    x = torch.randn((32, 20, 46))
    CTransformer = ChaoticTransformer(blocks, head, seqLen, input, output, dModel, dim, hidden, dropout, decoder, mish, chaotic, leeTanh, leeReLu, leeMish, chaosExtractor)
    output = CTransformer(x, plot = False)
    print(f'The output of the chaotic transformer (shape: {output.shape}):\n {output}')