'''
    Copyright:      JarvisLee
    Date:           10/22/2021
    File Name:      LeeOscillator.py
    Description:    The chaotic activation functions named Lee-Oscillator Based on Dr. Raymond Lee's paper.
'''

# Import the necessary library.
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Creat the class for the Mish activation function.
class Mish(nn.Module):
    # Create the constructor.
    def __init__(self):
        # Create the super constructor.
        super(Mish, self).__init__()
    # Create the forward.
    def forward(self, x):
        # Compute the mish.
        x = x * (torch.tanh(F.softplus(x)))
        # Return the output.
        return x

# Create the class for the Tanh based Lee-Oscillator.
class LeeTanh(nn.Module):
    '''
        The Tanh based Lee-Oscillator.\n
        Params:\n
            - a: The hyperparameters for Tanh based Lee-Oscillator.
            - K: The K coefficient of the Tanh based Lee-Oscillator.
            - N: The number of iterations of the Tanh based Lee-Oscillator.
    '''
    # Create the constructor.
    def __init__(self, a, K, N):
        # Create the super constructor.
        super(LeeTanh, self).__init__()
        # Get the hyperparameters for the Lee-Oscillator.
        self.a = a
        self.K = K
        self.N = N
    
    # Create the function to visualize the Tanh based Lee-Oscillator.
    def TanhCompute(self, filename):
        # Create the array to store and compute the value of the Tanh based Lee-Oscillator.
        u = torch.zeros([self.N])
        v = torch.zeros([self.N])
        z = torch.zeros([self.N])
        u[0] = 0.2
        z[0] = 0.2
        LeeTanh = np.zeros([1000, self.N])
        xAix = np.zeros([1000 * self.N])
        j = 0
        x = 0
        # Compute the Tanh based Lee-Oscillator.
        for i in np.arange(-1, 1, 0.002):
            for t in range(0, self.N - 1):
                u[t + 1] = torch.tanh(self.a[0] * u[t] - self.a[1] * v[t] + self.a[2] * z[t] + self.a[3] * i)
                v[t + 1] = torch.tanh(self.a[6] * z[t] - self.a[4] * u[t] - self.a[5] * v[t] + self.a[7] * i)
                w = torch.tanh(torch.Tensor([i]))
                z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-self.K * np.power(i, 2)) + w
                # Store the Lee-Oscillator.
                xAix[j] = i
                j = j + 1
                LeeTanh[x, t] = z[t + 1]
            LeeTanh[x, t + 1] = z[t + 1]
            x = x + 1
        # Store the Lee-Oscillator.
        plt.figure(1)
        fig = np.reshape(LeeTanh, [1000 * self.N])
        plt.plot(xAix, fig, ',')
        plt.savefig(f'./{filename}')
        plt.show()

    # Create the forward.
    def forward(self, x):
        # Get the random number.
        N = np.random.randint(1, self.N + 1)
        # Check the size of the x.
        if len(x.shape) > 2:
            # Initialize the internal and external states.
            u = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
            v = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
            # Initialize the output.
            z = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
        else:
            # Initialize the internal and external states.
            u = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
            v = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
            # Initialize the output.
            z = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
        # Adjust the internal states and the output.
        u[0] = u[0] + 0.2
        z[0] = z[0] + 0.2
        # Compute the forward.
        for t in range(0, N - 1):
            u[t + 1] = torch.tanh(self.a[0] * u[t] - self.a[1] * v[t] + self.a[2] * z[t] + self.a[3] * x)
            v[t + 1] = torch.tanh(self.a[6] * z[t] - self.a[4] * u[t] - self.a[5] * v[t] + self.a[7] * x)
            w = torch.tanh(x)
            z[t + 1] = (v[t + 1] - u[t + 1]) * torch.exp(-self.K * torch.pow(x, 2)) + w
        # Return the result.
        return z[-1]

# Create the class for the ReLu based Lee-Oscillator.
class LeeReLu(nn.Module):
    '''
        The ReLu based Lee-Oscillator.\n
        Params:\n
            - a: The hyperparameters for ReLu based Lee-Oscillator.
            - K: The K coefficient of the ReLu based Lee-Oscillator.
            - N: The number of iterations of the ReLu based Lee-Oscillator.
    '''
    # Create the constructor.
    def __init__(self, a, K, N):
        # Create the super constructor.
        super(LeeReLu, self).__init__()
        # Get the hyperparameters for the Lee-Oscillator.
        self.a = a
        self.K = K
        self.N = N
    
    # Create the function to visualize the ReLu based Lee-Oscillator.
    def ReLuCompute(self, filename):
        # Create the array to store and compute the value of the ReLu based Lee-Oscillator.
        u = torch.zeros([self.N])
        v = torch.zeros([self.N])
        z = torch.zeros([self.N])
        u[0] = 0.2
        z[0] = 0.2
        LeeReLu = np.zeros([1000, self.N])
        xAix = np.zeros([1000 * self.N])
        j = 0
        x = 0
        # Compute the ReLu based Lee-Oscillator.
        for i in np.arange(-1, 1, 0.002):
            for t in range(0, self.N - 1):
                u[t + 1] = torch.relu(self.a[0] * u[t] - self.a[1] * v[t] + self.a[2] * z[t] + self.a[3] * i)
                v[t + 1] = torch.relu(self.a[6] * z[t] - self.a[4] * u[t] - self.a[5] * v[t] + self.a[7] * i)
                w = torch.relu(torch.Tensor([i]))
                z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-self.K * np.power(i, 2)) + w
                # Store the Lee-Oscillator.
                xAix[j] = i
                j = j + 1
                LeeReLu[x, t] = z[t + 1]
            LeeReLu[x, t + 1] = z[t + 1]
            x = x + 1
        # Store the Lee-Oscillator.
        plt.figure(1)
        fig = np.reshape(LeeReLu, [1000 * self.N])
        plt.plot(xAix, fig, ',')
        plt.savefig(f'./{filename}')
        plt.show()

    # Create the forward.
    def forward(self, x):
        # Get the random number.
        N = np.random.randint(1, self.N + 1)
        # Check the size of the x.
        if len(x.shape) > 2:
            # Initialize the internal and external states.
            u = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
            v = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
            # Initialize the output.
            z = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
        else:
            # Initialize the internal and external states.
            u = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
            v = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
            # Initialize the output.
            z = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
        # Adjust the internal states and the output.
        u[0] = u[0] + 0.2
        z[0] = z[0] + 0.2
        # Compute the forward.
        for t in range(0, N - 1):
            u[t + 1] = torch.relu(self.a[0] * u[t] - self.a[1] * v[t] + self.a[2] * z[t] + self.a[3] * x)
            v[t + 1] = torch.relu(self.a[6] * z[t] - self.a[4] * u[t] - self.a[5] * v[t] + self.a[7] * x)
            w = torch.relu(x)
            z[t + 1] = (v[t + 1] - u[t + 1]) * torch.exp(-self.K * torch.pow(x, 2)) + w
        # Return the result.
        return z[-1]

# Create the class for the Mish based Lee-Oscillator.
class LeeMish(nn.Module):
    '''
        The Mish based Lee-Oscillator.\n
        Params:\n
            - a: The hyperparameters for Mish based Lee-Oscillator.
            - K: The K coefficient of the Mish based Lee-Oscillator.
            - N: The number of iterations of the Mish based Lee-Oscillator.
    '''
    # Create the constructor.
    def __init__(self, a, K, N):
        # Create the super constructor.
        super(LeeMish, self).__init__()
        # Get the hyperparameters for the Lee-Oscillator.
        self.a = a
        self.K = K
        self.N = N
        self.mish = Mish()
    
    # Create the function to visualize the Mish based Lee-Oscillator.
    def MishCompute(self, filename):
        # Create the array to store and compute the value of the Mish based Lee-Oscillator.
        u = torch.zeros([self.N])
        v = torch.zeros([self.N])
        z = torch.zeros([self.N])
        u[0] = 0.2
        z[0] = 0.2
        LeeMish = np.zeros([1000, self.N])
        xAix = np.zeros([1000 * self.N])
        j = 0
        x = 0
        # Compute the Mish based Lee-Oscillator.
        for i in np.arange(-1, 1, 0.002):
            for t in range(0, self.N - 1):
                u[t + 1] = self.mish(self.a[0] * u[t] - self.a[1] * v[t] + self.a[2] * z[t] + self.a[3] * i)
                v[t + 1] = self.mish(self.a[6] * z[t] - self.a[4] * u[t] - self.a[5] * v[t] + self.a[7] * i)
                w = self.mish(torch.Tensor([i]))
                z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-self.K * np.power(i, 2)) + w
                # Store the Lee-Oscillator.
                xAix[j] = i
                j = j + 1
                LeeMish[x, t] = z[t + 1]
            LeeMish[x, t + 1] = z[t + 1]
            x = x + 1
        # Store the Lee-Oscillator.
        plt.figure(1)
        fig = np.reshape(LeeMish, [1000 * self.N])
        plt.plot(xAix, fig, ',')
        plt.savefig(f'./{filename}')
        plt.show()

    # Create the forward.
    def forward(self, x):
        # Get the random number.
        N = np.random.randint(1, self.N + 1)
        # Check the size of the x.
        if len(x.shape) > 2:
            # Initialize the internal and external states.
            u = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
            v = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
            # Initialize the output.
            z = Variable(torch.zeros((N, x.shape[0], x.shape[1], x.shape[2]), dtype = torch.float32).to(x.device))
        else:
            # Initialize the internal and external states.
            u = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
            v = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
            # Initialize the output.
            z = Variable(torch.zeros((N, x.shape[0], x.shape[1]), dtype = torch.float32).to(x.device))
        # Adjust the internal states and the output.
        u[0] = u[0] + 0.2
        z[0] = z[0] + 0.2
        # Compute the forward.
        for t in range(0, N - 1):
            u[t + 1] = self.mish(self.a[0] * u[t] - self.a[1] * v[t] + self.a[2] * z[t] + self.a[3] * x)
            v[t + 1] = self.mish(self.a[6] * z[t] - self.a[4] * u[t] - self.a[5] * v[t] + self.a[7] * x)
            w = self.mish(x)
            z[t + 1] = (v[t + 1] - u[t + 1]) * torch.exp(-self.K * torch.pow(x, 2)) + w
        # Return the result.
        return z[-1]

# Test the codes.
if __name__ == "__main__":
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    x = nn.Parameter(torch.rand((5, 5)))
    Lee = LeeTanh(a, K, N)
    print(f"The original x: {x}")
    print(f"The x after the Tanh based Lee-Oscillator: {Lee(x)}")
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    Lee = LeeTanh(a, K, N)
    Lee.TanhCompute('LeeOscillator-A-Tanh.jpg')
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    K = 50
    N = 100
    Lee = LeeTanh(a, K, N)
    Lee.TanhCompute('LeeOscillator-B-Tanh.jpg')
    a = [0.55, 0.55, -0.5, 0.5, -0.55, -0.55, 0.5, -0.5]
    K = 50
    N = 100
    Lee = LeeTanh(a, K, N)
    Lee.TanhCompute('LeeOscillator-C-Tanh.jpg')
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    K = 300
    N = 100
    Lee = LeeTanh(a, K, N)
    Lee.TanhCompute('LeeOscillator-D-Tanh.jpg')
    a = [-0.2, 0.45, 0.6, 1, 0, -0.55, 0.55, 0]
    K = 100
    N = 100
    Lee = LeeTanh(a, K, N)
    Lee.TanhCompute('LeeOscillator-E-Tanh.jpg')

    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    x = nn.Parameter(torch.rand((5, 5)))
    Lee = LeeReLu(a, K, N)
    print(f"The original x: {x}")
    print(f"The x after the ReLu based Lee-Oscillator: {Lee(x)}")
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    Lee = LeeReLu(a, K, N)
    Lee.ReLuCompute('LeeOscillator-A-ReLu.jpg')
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    K = 50
    N = 100
    Lee = LeeReLu(a, K, N)
    Lee.ReLuCompute('LeeOscillator-B-ReLu.jpg')
    a = [0.55, 0.55, -0.5, 0.5, -0.55, -0.55, 0.5, -0.5]
    K = 50
    N = 100
    Lee = LeeReLu(a, K, N)
    Lee.ReLuCompute('LeeOscillator-C-ReLu.jpg')
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    K = 300
    N = 100
    Lee = LeeReLu(a, K, N)
    Lee.ReLuCompute('LeeOscillator-D-ReLu.jpg')
    a = [-0.2, 0.45, 0.6, 1, 0, -0.55, 0.55, 0]
    K = 100
    N = 100
    Lee = LeeReLu(a, K, N)
    Lee.ReLuCompute('LeeOscillator-E-ReLu.jpg')

    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    x = nn.Parameter(torch.rand((5, 5)))
    Lee = LeeMish(a, K, N)
    print(f"The original x: {x}")
    print(f"The x after the Mish based Lee-Oscillator: {Lee(x)}")
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    K = 50
    N = 100
    Lee = LeeMish(a, K, N)
    Lee.MishCompute('LeeOscillator-A-Mish.jpg')
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    K = 50
    N = 100
    Lee = LeeMish(a, K, N)
    Lee.MishCompute('LeeOscillator-B-Mish.jpg')
    a = [0.55, 0.55, -0.5, 0.5, -0.55, -0.55, 0.5, -0.5]
    K = 50
    N = 100
    Lee = LeeMish(a, K, N)
    Lee.MishCompute('LeeOscillator-C-Mish.jpg')
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    K = 300
    N = 100
    Lee = LeeMish(a, K, N)
    Lee.MishCompute('LeeOscillator-D-Mish.jpg')
    a = [-0.2, 0.45, 0.6, 1, 0, -0.55, 0.55, 0]
    K = 100
    N = 100
    Lee = LeeMish(a, K, N)
    Lee.MishCompute('LeeOscillator-E-Mish.jpg')