import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from common.utils import interp1d

class FNO2d(nn.Module):
    def __init__(self, pde, modes, width, input_size, output_size, domain):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.tw = input_size #time_window
        self.sys_dim = 2 #system dimension
        self.pde = pde
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2*input_size + 1, self.width)
        # input channel is 2*tw + 1: the solutions of the previous 2*tw timesteps + 1 location (u_1(t-tw, x), ..., u_1(t-1, x),u_2(t-tw, x), ..., u_2(t-1, x)  x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2*output_size)

        # domain = [a,b]
        self.domain = domain

    def forward(self, x):


        x = torch.flatten(x,1,2).permute(0, 2, 1) #[B,tw,2,n_x] --> [B,n_x,2*tw]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) #[B,n_x,2*tw] --> [B,n_x,2*tw + 1]
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        
        #[B, 2*tw, n_x] --> [B,tw,2,n_x]
        return x.unflatten(1,(self.tw, self.sys_dim))

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(self.domain[0], self.domain[1], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cdouble))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO2dParam(nn.Module):
    def __init__(self, pde, modes, width, input_size, output_size, domain, eq_variables):
        super(FNO2dParam, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.eq_variables = eq_variables #pde parameters to add to the network

        self.tw = input_size #time_window
        self.sys_dim = 2 #system dimension
        self.pde = pde
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2*input_size + 1 + len(eq_variables), self.width)
        # input channel is 2*tw + 1: the solutions of the previous 2*tw timesteps + 1 location (u_1(t-tw, x), ..., u_1(t-1, x),u_2(t-tw, x), ..., u_2(t-1, x)  x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2*output_size)

        # domain = [a,b]
        self.domain = domain

    def forward(self, x, variables):

        # Encode equation specific parameters
        variables_vec = []   
        if "a" in self.eq_variables.keys():
            variables_vec.append(variables["a"] / self.eq_variables["a"])
        if "b" in self.eq_variables.keys():
            variables_vec.append(variables["b"] / self.eq_variables["b"])
        #[n_param,B] --> [B,nx,n_param]
        variables_vec = torch.stack(variables_vec).permute(1,0).unsqueeze(1).repeat(1,x.size(-1),1).to(x.device)


        x = torch.flatten(x,1,2).permute(0, 2, 1) #[B,tw,2,n_x] --> [B,n_x,2*tw]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, variables_vec), dim=-1) #[B,n_x,2*tw] --> [B,n_x,2*tw + n_param]
        x = torch.cat((x, grid), dim=-1) #[B,n_x,2*tw + n_param] --> [B,n_x,2*tw + n_param + 1]
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        
        #[B, 2*tw, n_x] --> [B,tw,2,n_x]
        return x.unflatten(1,(self.tw, self.sys_dim))

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(self.domain[0], self.domain[1], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
class FNO2dParamUn(nn.Module):
    def __init__(self, pde, modes, width, input_size, output_size, domain, eq_variables):
        super(FNO2dParamUn, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.eq_variables = eq_variables #pde parameters to add to the network
        self.interp = True #to differentiate from non-interpolating FNOs

        self.tw = input_size #time_window
        self.sys_dim = 2 #system dimension
        self.pde = pde
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2*input_size + 1 + len(eq_variables), self.width)
        # input channel is 2*tw + 1: the solutions of the previous 2*tw timesteps + 1 location (u_1(t-tw, x), ..., u_1(t-1, x),u_2(t-tw, x), ..., u_2(t-1, x)  x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2*output_size)

        # domain = [a,b]
        self.domain = domain


    def three_to_two(self,x):
        # [n_batch, grid_size, n_features] --> [n_batch*n_features, grid_size]
        return x.permute(0,2,1).flatten(0,1)
    
    def two_to_three(self,x, nbatch, grid_size):
        # [n_batch*n_features, grid_size] --> [n_batch, grid_size, n_features]
        return x.unflatten(0,(nbatch,grid_size)).permute(0,2,1)


    def forward(self, x, variables, grid_unstruct):

        # Encode equation specific parameters
        variables_vec = []   
        if "a" in self.eq_variables.keys():
            variables_vec.append(variables["a"] / self.eq_variables["a"])
        if "b" in self.eq_variables.keys():
            variables_vec.append(variables["b"] / self.eq_variables["b"])
        #[n_param,B] --> [B,nx,n_param]
        variables_vec = torch.stack(variables_vec).permute(1,0).unsqueeze(1).repeat(1,x.size(-1),1).to(x.device)

        x = torch.flatten(x,1,2).permute(0, 2, 1) #[B,tw,2,n_x] --> [B,n_x,2*tw]

        #both grids in dim [B, n_x, 1]
        grid = self.get_grid(x.shape, x.device)
        grid_unstruct = grid_unstruct.unsqueeze(-1).to(x.device)

        nbatch, nfeat = x.size(0), x.size(-1)
        
        #[B, n_x, n_features]
        self.grid_unstruct_big = grid_unstruct.repeat(1,1,x.shape[-1])
        self.grid_big = grid.repeat(1,1,x.shape[-1])

        import matplotlib.pyplot as plt
        plt.plot(self.grid_unstruct_big[0,:,0].cpu(),x[0,:,0].cpu())

        #unstructured grid -> structured grid
        x = self.two_to_three(interp1d(self.three_to_two(self.grid_unstruct_big),self.three_to_two(x),self.three_to_two(self.grid_big)),nbatch,nfeat)

        x = torch.cat((x, variables_vec), dim=-1) #[B,n_x,2*tw] --> [B,n_x,2*tw + n_param]
        x = torch.cat((x, grid), dim=-1) #[B,n_x,2*tw + n_param] --> [B,n_x,2*tw + n_param + 1]
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        #structured grid -> unstructured grid
        #all input are [B, n_x, nfeat]
        x = self.two_to_three(interp1d(self.three_to_two(self.grid_big),self.three_to_two(x),self.three_to_two(self.grid_unstruct_big)),nbatch,nfeat)

        x = x.permute(0, 2, 1)
        
        #[B, 2*tw, n_x] --> [B,tw,2,n_x]
        return x.unflatten(1,(self.tw, self.sys_dim))

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(self.domain[0], self.domain[1], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
