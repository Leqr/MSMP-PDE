import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph
from equations.PDEs import *
from torch_geometric.utils.random import erdos_renyi_graph

#1D interpolation from https://raw.githubusercontent.com/al-jshen/spender/new-interp1d/spender/util.py
@torch.jit.script
def interp1d_single(
    x: torch.Tensor, y: torch.Tensor, target: torch.Tensor, mask: bool = True
) -> torch.Tensor:
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    b = y[:-1] - (m * x[:-1])

    idx = torch.sum(torch.ge(target[:, None], x[None, :]), 1) - 1
    idx = torch.clamp(idx, 0, len(m) - 1)

    itp = m[idx] * target + b[idx]

    if mask:
        low_mask = torch.le(target, x[0])
        high_mask = torch.ge(target, x[-1])
        itp[low_mask] = y[0]
        itp[high_mask] = y[-1]

    return itp


@torch.jit.script
def interp1d(
    x: torch.Tensor, y: torch.Tensor, target: torch.Tensor, mask: bool = True
) -> torch.Tensor:
    """One-dimensional linear interpolation. If x is not sorted, this will sort x for you.

    Args:
        x: the x-coordinates of the data points, must be increasing.
        y: the y-coordinates of the data points, same length as `x`.
        target: the x-coordinates at which to evaluate the interpolated values.
        mask: whether to clamp target values outside of the range of x (i.e., don't extrapolate)

    Returns:
        the interpolated values, same size as `target`.
    """
    # check dimensions
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    # check whether we need to broadcast x and y
    assert (
        x.shape[0] == y.shape[0] or x.shape[0] == 1 or y.shape[0] == 1
    ), f"x and y must have same length, or either x or y must have length 1, got {x.shape} and {y.shape}"

    if y.shape[0] == 1 and x.shape[0] > 1:
        y = y.expand(x.shape[0], -1)
        bs = x.shape[0]
    elif x.shape[0] == 1 and y.shape[0] > 1:
        x = x.expand(y.shape[0], -1)
        bs = y.shape[0]
    else:
        bs = x.shape[0]

    # check whether we need to broadcast target
    assert (
        target.shape[0] == bs or target.shape[0] == 1
    ), f"target must have same length as x and y, or length 1, got {target.shape} and {x.shape}"

    if target.shape[0] == 1:
        target = target.expand(bs, -1)

    # check for sorting
    if not torch.all(torch.diff(x, dim=-1) > 0):
        # if reverse-sorted, just flip
        if torch.all(torch.diff(x, dim=-1) < 0):
            x = x.flip(-1)
            y = y.flip(-1)
        else:
            # sort x and y if not already sorted
            x, idx = torch.sort(x, dim=-1)
            y = y[torch.arange(bs)[:, None], idx]

    # this is apparantly how parallelism works in pytorch?
    futures = [
        torch.jit.fork(interp1d_single, x[i], y[i], target[i], mask) for i in range(bs)
    ]
    itp = torch.stack([torch.jit.wait(f) for f in futures])

    return itp


class HDF5Dataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: PDE,
                 mode: str,
                 base_resolution: list=None,
                 super_resolution: list=None,
                 load_all: bool=False) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE ('CE' or 'WE')
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.dtype = torch.float64
        self.data = f[self.mode]
        self.base_resolution = (250, 100) if base_resolution is None else base_resolution
        self.super_resolution = (250, 200) if super_resolution is None else super_resolution
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'

        ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
        ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
        assert (ratio_nt.is_integer())
        assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)

        self.nt = self.data[self.dataset_base].attrs['nt']
        self.dt = self.data[self.dataset_base].attrs['dt']
        self.dx = self.data[self.dataset_base].attrs['dx']
        self.x = self.data[self.dataset_base].attrs['x']
        self.tmin = self.data[self.dataset_base].attrs['tmin']
        self.tmax = self.data[self.dataset_base].attrs['tmax']

        if load_all:
            data = {self.dataset_super: self.data[self.dataset_super][:]}
            f.close()
            self.data = data


    def __len__(self):
        return self.data[self.dataset_super].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        if(f'{self.pde}' == 'CE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1))
            weights = torch.tensor([[[[0.2]*5]]])
            u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            x = self.x

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['alpha'] = self.data['alpha'][idx]
            variables['beta'] = self.data['beta'][idx]
            variables['gamma'] = self.data['gamma'][idx]

            return u_base, u_super, x, variables
        
        elif(f'{self.pde}' == 'KF'):

            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            x = self.x
            left = torch.zeros_like(torch.tensor(u_super))[..., -3:-1]
            right = torch.zeros_like(torch.tensor(u_super))[..., 1:3]
            u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1))
            weights = torch.tensor([[[[0.2]*5]]])
            u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['r'] = self.data['r'][idx]
            variables['D'] = self.data['D'][idx]

            return u_base, u_super, x, variables
		
        elif(f'{self.pde}' == 'KS'):
            
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1))
            weights = torch.tensor([[[[0.2]*5]]])
            u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            x = self.x

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]

            return u_base, u_super, x, {}
    
        elif(f'{self.pde}' == 'WE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            # No padding is possible due to non-periodic boundary conditions
            weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            u_super = F.conv1d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # To match the downprojected trajectories, also coordinates need to be downprojected
            x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
            x = F.conv1d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['bc_left'] = self.data['bc_left'][idx]
            variables['bc_right'] = self.data['bc_right'][idx]
            variables['c'] = self.data['c'][idx]

            return u_base, u_super, x, variables

        elif(f'{self.pde}' == 'AD'):
            u_base = self.data[self.dataset_base][idx]
            
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, ...]
            x = self.x
            u_super = torch.tensor(u_super[...,0:-1:2]).squeeze().numpy()

            #separate case with unstructured grid
            if self.pde.untructured_grid == False:
                u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, ...]
                x = self.x
                u_super = torch.tensor(u_super[...,0:-1:2]).squeeze().numpy()
            else:
                u_super = u_base
                x = self.x

            variables = {}
            variables['a'] = self.data['a'][idx]
            variables['b'] = self.data['b'][idx]
            #u_super is of shape nt,2,nx
            return np.swapaxes(u_base,0,1), np.swapaxes(u_super,0,1), x, variables

        else:
            raise Exception("Wrong experiment")


class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE,
                 neighbors: int = 2,
                 time_window: int = 5,
                 t_resolution: int = 250,
                 x_resolution: int =100
                 ) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
            time_ration (int): temporal ratio between base and super resolution
            space_ration (int): spatial ratio between base and super resolution
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.t_res = t_resolution
        self.x_res = x_resolution

		#defines the probability to randomly create an edge in the uniform grid case
        self.random_probability = 0
        #self.random_probability = 1e-3

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)

    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels


    def create_graph(self,
                     data: torch.Tensor,
                     labels: torch.Tensor,
                     x: torch.Tensor,
                     variables: dict,
                     steps: list) -> Data:
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor
            variables (dict): dictionary of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)

        #periodic boundary condition --> cylindrical coordinate
        X = 2*np.pi*x[0]/(torch.max(x[0])-1e-3)
        x_per = torch.zeros(len(X),2)
        x_per[:,0] = torch.cos(X)
        x_per[:,1] = torch.sin(X)

        u, x_pos, x_pos_per, t_pos, y, batch = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):
            if f'{self.pde}' == 'AD':
                d = data_batch.transpose(0,1).flatten(0,1)
                u = torch.cat((u,torch.transpose(d,0,1)),)
                l = labels_batch.transpose(0,1).flatten(0,1)
                y = torch.cat((y,torch.transpose(l,0,1)),)
            else:
                u = torch.cat((u, torch.transpose(torch.cat([d[None, :] for d in data_batch]), 0, 1)), )
                y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )

            x_pos = torch.cat((x_pos, x[0]), )
            x_pos_per = torch.cat((x_pos_per, x_per), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
            batch = torch.cat((batch, torch.ones(nx) * b), )

        # Calculate the edge_index
        if f'{self.pde}' == 'CE' or f'{self.pde}' == 'KF' or f'{self.pde}' == 'KS' or f'{self.pde}' == 'AD':
            dx = x[0][1] - x[0][0]
            radius = self.n * dx + 0.0001
            edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
            if self.random_probability > 0:
                #random edge sampling
                edge_index_random = erdos_renyi_graph(len(x_pos),self.random_probability)
                #merge edge_index
                edge_index = torch.unique(torch.cat((edge_index,edge_index_random),dim = 1),dim=1)
            
            #handle the AD case with unstructured grid --> knn graph generation
            if f'{self.pde}' == 'AD' and self.pde.untructured_grid:
                edge_index = knn_graph(x_pos_per, k=self.n, batch=batch.long(), loop=False)

        elif f'{self.pde}' == 'WE':
            edge_index = knn_graph(x_pos, k=self.n, batch=batch.long(), loop=False)

        graph = Data(x=u, edge_index=edge_index)
        graph.y = y
        graph.pos = torch.cat((t_pos[:, None], x_pos[:, None]), 1)
        graph.batch = batch.long()

        # Equation specific parameters
        if f'{self.pde}' == 'CE':
            alpha, beta, gamma = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i in batch.long():
                alpha = torch.cat((alpha, torch.tensor([variables['alpha'][i]])[:, None]), )
                beta = torch.cat((beta, torch.tensor([variables['beta'][i]*(-1.)])[:, None]), )
                gamma = torch.cat((gamma, torch.tensor([variables['gamma'][i]])[:, None]), )

            graph.alpha = alpha
            graph.beta = beta
            graph.gamma = gamma
        
        elif f'{self.pde}' == 'KF':
            r, D = torch.Tensor(), torch.Tensor()
            for i in batch.long():
                r = torch.cat((r, torch.tensor([variables['r'][i]])[:, None]), )
                D = torch.cat((D, torch.tensor([variables['D'][i]])[:, None]), )

            graph.r = r
            graph.D = D

        elif f'{self.pde}' == 'WE':
            bc_left, bc_right, c = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i in batch.long():
                bc_left = torch.cat((bc_left, torch.tensor([variables['bc_left'][i]])[:, None]), )
                bc_right = torch.cat((bc_right, torch.tensor([variables['bc_right'][i]])[:, None]), )
                c = torch.cat((c, torch.tensor([variables['c'][i]])[:, None]), )

            graph.bc_left = bc_left
            graph.bc_right = bc_right
            graph.c = c

        elif f'{self.pde}' == 'AD':
            a, b = torch.Tensor(), torch.Tensor()
            for i in batch.long():
                a = torch.cat((a, torch.tensor([variables['a'][i]])[:,None]), )
                b = torch.cat((b, torch.tensor([variables['b'][i]])[:, None]), )
 
            graph.a = a
            graph.b = b

        return graph


    def create_next_graph(self,
                             graph: Data,
                             pred: torch.Tensor,
                             labels: torch.Tensor,
                             steps: list) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # Output is the new input
        graph.x = torch.cat((graph.x, pred), 1)
        if f'{self.pde}' == 'AD':
            graph.x = graph.x[:, 2*self.tw:]
        else :
            graph.x = graph.x[:, self.tw:]

        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        # Update labels and input timesteps
        y, t_pos = torch.Tensor(), torch.Tensor()
        for (labels_batch, step) in zip(labels, steps):
            if f'{self.pde}' == 'AD':
                #2d Solution
                #collapse the sol dim on the time_window
                l = labels_batch.transpose(0,1).flatten(0,1)
                y = torch.cat((y,torch.transpose(l,0,1)),)
            else :
                y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
        graph.y = y
        graph.pos[:, 0] = t_pos

        return graph

