from torch.nn import functional as F
from torch import nn
import torch
from experiments.models_gnn2D import unflatten_u
from equations.PDEs import PDE

class BaseCNN2D(nn.Module):
    '''
    A simple baseline 1D Res CNN approach, the time dimension is stacked in the channels
    '''
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_channels: int = 128,
                 padding_mode: str = f'circular') -> None:
        """
        Initialize the simple CNN architecture. It contains 8 1D CNN-layers with skip connections
        and increasing receptive field.
        The input to the forward pass has the shape [batch, time_window, 2, x].
        The output has the shape [batch, time_window, 2, x].
        Args:
            pde (PDE): the PDE at hand
            time_window (int): input/output timesteps of the trajectory
            hidden_channels: hidden channel dimension
            padding_mode (str): circular mode as default for periodic boundary problems
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.time_window = time_window
        self.hidden_channels = hidden_channels
        self.padding_mode = padding_mode

        self.conv1 = nn.Conv1d(in_channels=2*self.time_window, out_channels=self.hidden_channels, kernel_size=3, padding=1,
                               padding_mode=self.padding_mode, bias=True)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=5, padding=2,
                               padding_mode=self.padding_mode, bias=True)
        self.conv3 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=5, padding=2,
                               padding_mode=self.padding_mode, bias=True)
        self.conv4 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=5, padding=2,
                               padding_mode=self.padding_mode, bias=True)
        self.conv5 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=7, padding=3,
                               padding_mode=self.padding_mode, bias=True)
        self.conv6 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=7, padding=3,
                               padding_mode=self.padding_mode, bias=True)
        self.conv7 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=7, padding=3,
                               padding_mode=self.padding_mode, bias=True)
        self.conv8 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=2*self.time_window, kernel_size=9, padding=4,
                               padding_mode=self.padding_mode, bias=True)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)



    def __repr__(self):
        return f'BaseCNN'

    def forward(self, u):
        """Forward pass of solver
        """

        u_in = torch.flatten(u,1,2)

        x = F.elu(self.conv1(u_in))
        x = x + F.elu(self.conv2(x))
        x = x + F.elu(self.conv3(x))
        x = x + F.elu(self.conv4(x))
        x = x + F.elu(self.conv5(x))
        x = x + F.elu(self.conv6(x))
        x = x + F.elu(self.conv7(x))
        diff = self.conv8(x).unflatten(1,(self.time_window,2))

        dt = (torch.ones(1, self.time_window, 1) * self.pde.dt).to(diff.device)
        dt = torch.cumsum(dt, dim=1).unsqueeze(-1)

        out = u + dt * diff

        return out
