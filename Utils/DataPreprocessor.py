'''
    Copyright:      JarvisLee
    Date:           10/21/2021
    Filename:       DataPreprocessor.py
    Description:    This file is used to preprocess the training data.
'''

# Import the necessary library.
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from Utils.ParamsHandler import Handler

# Get the hyperparameters' handler.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))

# Set the class to encapsulate the dataset generator's tools.
class GetDataset(torch.utils.data.Dataset):
    '''
        This class is used to encapsulate all the functions to generate the dataset.\n
        This class contains three parts:\n
            - '__init__' is used to get the raw data and raw target.
            - '__getitem__' is used to get each data and each target.
            - '__len__' is used to get the length of each data.
    '''
    # Create the constructor.
    def __init__(self, rawData, rawTarget):
        # Create the super constructor.
        super(GetDataset, self).__init__()
        # Get the data.
        self.data = rawData
        self.target = rawTarget
    
    # Create the function to get the index.
    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        # Return the data and target.
        return data, target
    
    # Create the function to get the data length.
    def __len__(self):
        return len(self.data)

# Set the class to encapsulate data preprocessor's tools.
class Preprocessor():
    '''
        This class is used to encapsulate all the functions which is used to preprocess the dataset.\n
        This class contains two parts:\n
            - 'Normalization' is used to do the normalization of the raw data.
            - 'FXTrainData' is used to preprocess the training data and development data.
    '''
    # Set the function to compute the normalization of the raw data.
    def Normalization(x, ymin = -1, ymax = 1):
        '''
            This function is used to compute the normalization of the raw data.\n
            Params:\n
                - x: The raw data.
                - ymin: The minimum standard of the normalization.
                - ymax: The maximum standard of the normalization.
        '''
        # Get the xmax and xmin.
        xmax = np.max(x, axis = 0)
        xmin = np.min(x, axis = 0)
        # Normalize the raw data.
        output = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
        # Return the output.
        return output, xmax, xmin
    
    # Set the function to preprocess the training data and development data.
    def FXTrainData(dataDir, batchSize, trainPercent, shuffle = True):
        '''
            This function is used to preprocess the training data and development data.\n
            Params:\n
                - dataDir: The directary of the raw data.
                - batchSize: The batch size of the datasets.
                - trainPercent: The percentage of the raw data which are the training data.
                - shuffle: The boolean to check whether shuffle the raw data.
        '''
        # Set the list to store the training data.
        data = []
        target = []
        # Get all the files.
        for filename in os.listdir(dataDir):
            # Read the data in each file.
            raw = np.array(pd.read_csv(dataDir + "//" + filename, index_col = (0)).values)
            # Check whether do the normalization or not.
            if Cfg.Normalize:
                # Normalize the data.
                raw, max, min = Preprocessor.Normalization(raw)
                # Store the max and min to do the denormalization.
                if not os.path.exists(".//FXTrade//FXMax") or not os.path.exists(".//FXTrade//FXMin"):
                    os.mkdir(".//FXTrade//FXMax")
                    os.mkdir(".//FXTrade//FXMin")
                norm = pd.DataFrame(max)
                norm.to_csv(".//FXTrade//FXMax" + f"//{filename}", index = None, header = None)
                norm = pd.DataFrame(min)
                norm.to_csv(".//FXTrade//FXMin" + f"//{filename}", index = None, header = None)
            # Split the raw data into dataset.
            for i in range(0, raw.shape[0]):
                # Check whether there are still enough data are remained.
                if (raw.shape[0] - i >= Cfg.Days + 1):
                    # Get the raw data.
                    rawData = raw[i:(i + Cfg.Days), :]
                    rawTarget = raw[(i + Cfg.Days):(i + Cfg.Days + 1), :Cfg.labels]
                    # Add the data into the data and target.
                    data.append(rawData)
                    target.append(rawTarget.T)
            # Give the hint for completing reading one file's data.
            print(f"{filename}'s data reading is completed!")
        # Shuffle the raw data.
        if shuffle:
            dataIndex = []
            for i in range(len(data)):
                dataIndex.append(i)
            np.random.shuffle(dataIndex)
            # Rearrange the data.
            tempData = []
            tempTarget = []
            for each in dataIndex:
                tempData.append(data[each])
                tempTarget.append(target[each])
            data = tempData
            target = tempTarget
        # Convert the list to be the tensor.
        data = torch.tensor(np.array(data), dtype = torch.float32)
        target = torch.tensor(np.array(target), dtype = torch.float32).squeeze()
        # Get the training data boundary.
        bound = int(data.shape[0] * trainPercent)
        # Generate the datasets.
        trainSet = GetDataset(data[:bound, :, :], target[:bound, :])
        devSet = GetDataset(data[bound:, :, :], target[bound:, :])
        # Get the training data and development data.
        trainData = DataLoader(dataset = trainSet, batch_size = batchSize, shuffle = shuffle, drop_last = False)
        devData = DataLoader(dataset = devSet, batch_size = batchSize, shuffle = False, drop_last = False)
        # Return the training data and development data.
        return trainData, devData

# Test the codes.
if __name__ == "__main__":
    trainData, devData = Preprocessor.FXTrainData(Cfg.dataDir, Cfg.batchSize, 0.9)
    print(f"The length of the train data {len(trainData)}")
    print(f"The length of the dev data {len(devData)}")
    for i, (data, target) in enumerate(trainData):
        print(f"Data {i + 1}: shape: {data.shape}, {target.shape}")
        print(data)
        print(target)
    for i, (data, target) in enumerate(devData):
        print(f"Data {i + 1}: shape: {data.shape}, {target.shape}")
        print(data)
        print(target)
    rawData = torch.randn(100, 10, 10)
    rawTarget = torch.randint(0, 2, (100, 1))
    data = GetDataset(rawData, rawTarget)
    trainData = DataLoader(data, 32, shuffle = False, drop_last = False)
    for i, (data, target) in enumerate(trainData):
        print(f"Data {i + 1}: shape: {data.shape}, {target.shape}")