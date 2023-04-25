import torch
from torch import nn
from torch_geometric.data import Data
from equations.PDEs import *
from experiments.models_gnn import Swish, GNN_Layer, GNN_LayerLin, LEM, LSTM, LEMS
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv, RGATConv
from torch_scatter import scatter

def unflatten_u(u : torch.Tensor, time_window : float):
    """Unflatten the merged sols dimensions
        [*,n*time_window] --> [*,n,time_window]
    """
    dim = 1
    return u.unflatten(dim,(u.size(dim)//time_window, time_window))


class MP_PDE_Solver2D(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver2D, self).__init__()
        # 1D decoder CNN is so far designed time_window = [25,50]
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(2*self.time_window + 2 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )

        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 2, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos_x, variables), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=2)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp(h)
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)

        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = u_unf + dt * diff

        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

class MP_PDE_Solver2DGated(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver2DGated, self).__init__()
        # 1D decoder CNN is so far designed time_window = 25,50
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))


        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )


        self.embedding_mlp = nn.Sequential(
            nn.Linear(2*self.time_window + 2 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        self.swish = Swish()

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )

        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 2, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos_x, variables), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=2)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp(h)
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)

        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = u_unf + dt * diff

        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

class MP_PDE_Solver2DLEMLinGated(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {},
                 save_state = None
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver2DLEMLinGated, self).__init__()
        # 1D decoder CNN is so far designed time_window = 25,50
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        #select LEM or LEMS with state saving
        self.save_state = save_state

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))


        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        #selector for LEM mode
        if self.save_state is None:
            self.embedding_lem = LEM
        elif self.save_state:
             self.embedding_lem = LEMS
            
        self.embedding_lem = self.embedding_lem(2+len(self.eq_variables)+2,self.hidden_features)

        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )


        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 2, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(data.x.device)
        dt = torch.cumsum(dt, dim=2)

        #create t array for the LEM
        ts = (dt + pos_t).squeeze()

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        #get and array of size [time_window,n_batch,5] containing [x_i,u^1_{i,j},u^2_{i,j},t_j,theta]
        lem_input_full = torch.empty(self.time_window,lem_input.size(1),2+len(self.eq_variables)+2).to(device = node_input.device)
        for i in range(self.time_window):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],lem_input[i + self.time_window],ts[:,i].unsqueeze(-1),variables[:,1:]),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)
        
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp(h)
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)
        
        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = u_unf + dt * diff
        
        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

class MP_PDE_Solver2DLEMLinG2(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver2DLEMLinG2, self).__init__()
        # 1D decoder CNN is so far designed time_window = 25, 50
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))


        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )


        self.embedding_lem = LEM(2+len(self.eq_variables)+2,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )


        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 2, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(data.x.device)
        dt = torch.cumsum(dt, dim=2)

        #create t array for the LEM
        ts = (dt + pos_t).squeeze()

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        #get and array of size [time_window,n_batch,5] containing [x_i,u^1_{i,j},u^2_{i,j},t_j,theta]
        lem_input_full = torch.empty(self.time_window,lem_input.size(1),2+len(self.eq_variables)+2).to(device = node_input.device)
        for i in range(self.time_window):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],lem_input[i + self.time_window],ts[:,i].unsqueeze(-1),variables[:,1:]),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)
        
        for i in range(self.hidden_layer):
            tau = self.swish(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau = torch.tanh(scatter((torch.abs(tau[edge_index[0]] - tau[edge_index[1]]) ** 2).squeeze(-1),
                                 edge_index[0], 0,dim_size=tau.size(0), reduce='mean'))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp(h)
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)
        
        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = u_unf + dt * diff
        
        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

class MP_PDE_Solver2DLSTMLinGated(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver2DLSTMLinGated, self).__init__()
        # 1D decoder CNN is so far designed time_window = 25, 50
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))


        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )


        self.embedding_lstm = LSTM(2+len(self.eq_variables)+2,self.hidden_features)
        self.lstmoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )


        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 2, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(data.x.device)
        dt = torch.cumsum(dt, dim=2)

        #create t array for the LEM
        ts = (dt + pos_t).squeeze()

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        #get and array of size [time_window,n_batch,5] containing [x_i,u^1_{i,j},u^2_{i,j},t_j,theta]
        lem_input_full = torch.empty(self.time_window,lem_input.size(1),2+len(self.eq_variables)+2).to(device = node_input.device)
        for i in range(self.time_window):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],lem_input[i + self.time_window],ts[:,i].unsqueeze(-1),variables[:,1:]),-1)

        h = self.embedding_lstm(lem_input_full)
        h = self.lstmoutput_mlp(h)
        
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp(h)
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)
        
        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = u_unf + dt * diff
        
        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

class MP_PDE_Solver2DLSTMLin(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver2DLSTMLin, self).__init__()
        # 1D decoder CNN is so far designed time_window = [25, 50]
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_lstm = LSTM(2+len(self.eq_variables)+2,self.hidden_features)
        self.lstmoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )

        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 2, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(data.x.device)
        dt = torch.cumsum(dt, dim=2)

        #create t array for the LEM
        ts = (dt + pos_t).squeeze()

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        #get and array of size [time_window,n_batch,5] containing [x_i,u^1_{i,j},u^2_{i,j},t_j,theta]
        lem_input_full = torch.empty(self.time_window,lem_input.size(1),2+len(self.eq_variables)+2).to(device = node_input.device)
        for i in range(self.time_window):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],lem_input[i + self.time_window],ts[:,i].unsqueeze(-1),variables[:,1:]),-1)

        h = self.embedding_lstm(lem_input_full)
        h = self.lstmoutput_mlp(h)

        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp(h)
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)

        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = u_unf + dt * diff

        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

class MP_PDE_Solver2DLEMLin(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver2DLEMLin, self).__init__()
        # 1D decoder CNN is so far designed time_window = [25, 50]
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_lem = LEM(2+len(self.eq_variables)+2,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )

        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 2, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(data.x.device)
        dt = torch.cumsum(dt, dim=2)

        #create t array for the LEM
        ts = (dt + pos_t).squeeze()

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        #get and array of size [time_window,n_batch,5] containing [x_i,u^1_{i,j},u^2_{i,j},t_j,theta]
        lem_input_full = torch.empty(self.time_window,lem_input.size(1),2+len(self.eq_variables)+2).to(device = node_input.device)
        for i in range(self.time_window):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],lem_input[i + self.time_window],ts[:,i].unsqueeze(-1),variables[:,1:]),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)

        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp(h)
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)

        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = u_unf + dt * diff

        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

class G_PDE_Solver2DLEMLinGated(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(G_PDE_Solver2DLEMLinGated, self).__init__()
        # 1D decoder CNN is so far designed time_window = 25, 50
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        graph = RGATConv

        self.gnn_layers = torch.nn.ModuleList(modules=(graph(self.hidden_features,
                                                                self.hidden_features
        ,edge_dim=51) for _ in range(self.hidden_layer)))

        self.gnn_layers_gate = torch.nn.ModuleList(modules=(graph(self.hidden_features,
                                                                self.hidden_features
        ,edge_dim=51) for _ in range(self.hidden_layer)))

        self.embedding_lem = LEM(2+len(self.eq_variables)+2,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )


        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 2, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(2, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch
    
        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(data.x.device)
        dt = torch.cumsum(dt, dim=2)

        #create t array for the LEM
        ts = (dt + pos_t).squeeze()

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        #get and array of size [time_window,n_batch,5] containing [x_i,u^1_{i,j},u^2_{i,j},t_j,theta]
        lem_input_full = torch.empty(self.time_window,lem_input.size(1),2+len(self.eq_variables)+2).to(device = node_input.device)

        for i in range(self.time_window):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],lem_input[i + self.time_window],ts[:,i].unsqueeze(-1),variables[:,1:]),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)

        #compute the edge vector as a finite difference term
        edge_attr = torch.cat((u[edge_index[0]] - u[edge_index[1]],pos_x[edge_index[0]] - pos_x[edge_index[1]]),-1)
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h,edge_index,edge_attr))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h,edge_index,edge_attr))

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp(h)
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)
        
        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = u_unf + dt * diff
        
        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

class MP_PDE_Solver2DLEMLinGatedGLU(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 164,
                 hidden_layer: int = 6,
                 eq_variables: dict = {},
                 save_state = None
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver2DLEMLinGatedGLU, self).__init__()
        # 1D decoder CNN is so far designed time_window = 25,50
        assert(time_window == 25 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        #select LEM or LEMS with state saving
        self.save_state = save_state

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))


        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=2*self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=2*self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        #selector for LEM mode
        if self.save_state is None:
            self.embedding_lem = LEM
        elif self.save_state:
             self.embedding_lem = LEMS
            
        self.embedding_lem = self.embedding_lem(2+len(self.eq_variables)+2,self.hidden_features)

        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        self.double_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, 2*self.hidden_features),
            Swish(),
            nn.Unflatten(1,(2,self.hidden_features))
        )


        if (self.time_window == 25):
            self.output_mlp_diff = nn.Sequential(nn.Conv1d(2, 8, 6, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 15, stride=1)
                                            )
            self.output_mlp_gate = nn.Sequential(nn.Conv1d(2, 8, 6, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 2, 15, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        variables = pos_t    # time is treated as equation variable
        if "a" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["a"]), -1)
        if "b" in self.eq_variables.keys():
            variables = torch.cat((variables, data.a / self.eq_variables["b"]), -1)

        dt = (torch.ones(1, 1, self.time_window) * self.pde.dt).to(data.x.device)
        dt = torch.cumsum(dt, dim=2)

        #create t array for the LEM
        ts = (dt + pos_t).squeeze()

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        #get and array of size [time_window,n_batch,5] containing [x_i,u^1_{i,j},u^2_{i,j},t_j,theta]
        lem_input_full = torch.empty(self.time_window,lem_input.size(1),2+len(self.eq_variables)+2).to(device = node_input.device)
        for i in range(self.time_window):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],lem_input[i + self.time_window],ts[:,i].unsqueeze(-1),variables[:,1:]),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)
        
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        #[n_batch*n_nodes, hidden_dim] -> [n_batch*n_nodes, 2*hidden_dim] --> [n_batch*n_nodes, 2, hidden_dim] 
        h = self.double_mlp(h)

        # Decoder (formula 10 in the paper)
        # [batch*n_nodes, 2, hidden_dim] -> 1DCNN([batch*n_nodes, 2, hidden_dim]) -> [batch*n_nodes, 2, time_window]
        diff = self.output_mlp_diff(h[:,:,h.size(2)//2:])
        scale = self.output_mlp_gate(h[:,:,:h.size(2)//2])
        #unflatten the array to timestep the function values of both sols
        #[batch*n_nodes, 2*time_window] -> [batch*n_nodes, 2 time_window]
        u_unf = unflatten_u(u,self.time_window)
        
        #[batch*n_nodes, 2, time_window] + [1, 1, time_window]*[batch*n_nodes, 2, time_window] = 
        #[batch*n_nodes, 2, time_window]
        out = (torch.ones_like(scale)-scale)*u_unf + dt * scale * diff
        
        #need to flatten again to be in the same shape as input data ([batch*n_nodes, 2*time_window])
        return torch.flatten(out,1,2)

