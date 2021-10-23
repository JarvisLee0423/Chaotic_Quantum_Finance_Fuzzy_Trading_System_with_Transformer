'''
    Copyright:      JarvisLee
    Date:           10/24/2021
    File Name:      NoamScheduler.py
    Description:    This file is used to generate the naom learning rate scheduler in the paper 'Attention is all you need'.          
'''

# Import the necessary library.
import torch.optim as optim
from torchvision import models

# Set the class to encapsulate the noam scheduler.
class NoamScheduler():
    '''
        This class is used to encapsulate the noam learning rate scheduler.\n
        Params:\n
            - optimizer: The optimization method of the model training.
            - warmUp: The total learning rate warm up steps.
            - dModel: The dimension of the input data.
    '''
    # Create the constructor.
    def __init__(self, optimizer, warmUp, dModel):
        # Set the member variables.
        self.optim = optimizer
        self.warmUp = warmUp
        self.dModel = dModel
        self.stepNum = 0
    
    # Create the class to forward the step of the scheduler.
    def step(self):
        # Get the current step.
        self.stepNum = self.stepNum + 1
        # Compute the current learning rate.
        clr = (self.dModel ** (-0.5)) * min(self.stepNum ** (-0.5), self.stepNum * self.warmUp ** (-1.5))
        # Update the learning rate in the optimizer.
        for params in self.optim.param_groups:
            params['lr'] = clr

# Test the codes.
if __name__ == "__main__":
    model = models.AlexNet()
    optimizer = optim.Adam(model.parameters(), lr = 0.01, betas = (0.9, 0.98), eps = 1e-09)
    scheduler = NoamScheduler(optimizer, 4000, 100)
    for i in range(1, 51701):
        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])