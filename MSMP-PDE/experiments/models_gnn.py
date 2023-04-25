import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from torch_geometric.data import Data
from equations.PDEs import *
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm
import lem_cuda
from torch_scatter import scatter


class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)

class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 1 + n_variables, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update

class GNN_LayerLin(MessagePassing):
    """
    Message passing layer without final activation function
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_LayerLin, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 1 + n_variables, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features))
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return update
        else:
            return update

class MP_PDE_Solver(torch.nn.Module):
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
        super(MP_PDE_Solver, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window + 2 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )


        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos_x, variables), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out
    
########## LEM ##########

class LEMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, weights_lin_z, bias,
                bias_lin_z, initial_y_state, initial_z_state, dt):
        all_y_states, all_z_states, all_X, all_X2, \
        all_multi_scales, all_lin_new_z_state = lem_cuda.forward(inputs, weights,
                                                                 weights_lin_z, bias, bias_lin_z,
                                                                 initial_y_state, initial_z_state, dt)
        ctx.save_for_backward(all_X, all_X2, all_multi_scales,
                              all_lin_new_z_state, weights, weights_lin_z, bias,
                              bias_lin_z, initial_y_state, initial_z_state, dt)
        return all_y_states, all_z_states

    @staticmethod
    def backward(ctx, grad_y_states, grad_z_states):
        outputs = lem_cuda.backward(grad_y_states.contiguous(), grad_z_states.contiguous(), *ctx.saved_tensors)
        d_inputs, d_weights, d_weights_lin_z, d_bias, d_bias_lin_z, d_initial_y_state, d_initial_z_state = outputs
        return None, d_weights, d_weights_lin_z, d_bias, d_bias_lin_z, d_initial_y_state, d_initial_z_state, None


class LEMcuda(torch.nn.Module):
    def __init__(self, ninp, nhid, dt):
        super(LEMcuda, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.weights = torch.nn.Parameter(torch.empty(3 * nhid, ninp + nhid))
        self.weights_lin_z = torch.nn.Parameter(torch.empty(nhid, ninp + nhid))
        self.bias = torch.nn.Parameter(torch.empty(3 * nhid))
        self.bias_lin_z = torch.nn.Parameter(torch.empty(nhid))
        self.dt = torch.tensor(dt).reshape(1, 1).cuda()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, states=None):
        if states is None:
            y = input.data.new(input.size(1), self.nhid).zero_()
            z = input.data.new(input.size(1), self.nhid).zero_()
            states = (y,z)
        return LEMFunction.apply(input.contiguous(), self.weights,
                                 self.weights_lin_z, self.bias,
                                 self.bias_lin_z, *states, self.dt)


class LEM(torch.nn.Module):
    def __init__(self, ninp, nhid, dt=1.):
        super(LEM, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.rnn = LEMcuda(ninp,nhid,dt)

    def forward(self, input):
        all_y, all_z = self.rnn(input)
        return all_y[-1]
    
    
class LEMS(torch.nn.Module):
    #stores the hidden state for the following call
    def __init__(self, ninp, nhid, dt=1.):
        super(LEMS, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.rnn = LEMcuda(ninp,nhid,dt)
        self.states = None

    def forward(self, input):
        all_y, all_z = self.rnn(input, self.states)
        self.states = (all_y[-1], all_z[-1])
        return all_y[-1]
    
    def reset_states(self):
        #reset the states for starting a new unrollings sequence
        self.states = None

############# Models #############

class MP_PDE_SolverLEM(torch.nn.Module):
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
        super(MP_PDE_SolverLEM, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )
        self.save_state = False
        self.embedding_lem = LEM(2+len(self.eq_variables)+1,self.hidden_features)



        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lem(lem_input_full)

        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out

class LEMLin(torch.nn.Module):
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
        super(LEMLin, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.embedding_lem = LEM(2+len(self.eq_variables)+1,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())



        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out

class MP_PDE_SolverLEMLin(torch.nn.Module):
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
        super(MP_PDE_SolverLEMLin, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )
        self.embedding_lem = LEM(2+len(self.eq_variables)+1,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())



        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)

        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out

class LSTM(torch.nn.Module):
    def __init__(self, ninp, nhid):
        super(LSTM, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.rnn = nn.LSTM(ninp,nhid)

    def forward(self, input):
        output, (hn, cn) = self.rnn(input)
        return output[-1]


class MP_PDE_SolverLSTMLin(torch.nn.Module):
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
        super(MP_PDE_SolverLSTMLin, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_lstm = LSTM(2+len(self.eq_variables)+1,self.hidden_features)
        self.lstmoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())


        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)

        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lstm(lem_input_full)
        h = self.lstmoutput_mlp(h)

        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out

class MP_PDE_SolverLSTMLinGated(torch.nn.Module):
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
        super(MP_PDE_SolverLSTMLinGated, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
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
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))
        
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_lstm = LSTM(2+len(self.eq_variables)+1,self.hidden_features)
        self.lstmoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)


        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lstm(lem_input_full)
        h = self.lstmoutput_mlp(h)
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out

class MP_PDE_SolverGated(torch.nn.Module):
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
        super(MP_PDE_SolverGated, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
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
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))
        
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window + 2 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        self.swish = Swish()

        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos_x, variables), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))
            #Mod5
            #h = h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out

class MP_PDE_SolverLEMLinGated(torch.nn.Module):
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
        super(MP_PDE_SolverLEMLinGated, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
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
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))
        
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_lem = LEM(2+len(self.eq_variables)+1,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)


        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)
        
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out

class MP_PDE_SolverLEMLinGatedGLU(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 164,
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
        super(MP_PDE_SolverLEMLinGatedGLU, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
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
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))
        
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_lem = LEM(2+len(self.eq_variables)+1,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        # Decoder CNN, maps to different outputs (temporal bundling)
        self.output_mlp_gate = nn.Sequential(nn.Conv1d(1, 8, 6, stride=2),Swish(),nn.Conv1d(8, 1, 15, stride=1))
        self.output_mlp_diff = nn.Sequential(nn.Conv1d(1, 8, 6, stride=2),Swish(),nn.Conv1d(8, 1, 15, stride=1))
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)


        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)
        
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))
        scale = self.output_mlp_gate(h[:,:h.size(1)//2][:,None]).squeeze(1)
        diff = self.output_mlp_diff(h[:,h.size(1)//2:][:,None]).squeeze(1)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        
        out = (torch.ones_like(scale) - scale)*u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * (scale * diff)

        return out
        
class MSSMP_PDE_Solver_sub(torch.nn.Module):
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
        super(MSSMP_PDE_Solver_sub, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
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
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))
        
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_lem = LEM(2+len(self.eq_variables)+1,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)


        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)
        
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        #out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return diff
    
class MSSMP_PDE_Solver(torch.nn.Module):
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
        super(MSSMP_PDE_Solver, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        self.diff = MSSMP_PDE_Solver_sub(pde,time_window,hidden_features,hidden_layer,eq_variables)
        self.scale = MSSMP_PDE_Solver_sub(pde,time_window,hidden_features,hidden_layer,eq_variables)

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

        scale = self.scale(data)
        diff = self.diff(data)

        u = data.x
        
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(data.x.device)
        dt = torch.cumsum(dt, dim=1)

        #wrong implementation# out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1)*scale + dt * (scale * diff)
        out = (torch.ones_like(scale) - scale)*u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * (scale * diff)

        return out
    
class MP_PDE_SolverLEMLinGatedSave(torch.nn.Module):
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
        super(MP_PDE_SolverLEMLinGatedSave, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
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
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))
        
        self.gnn_layers_gate = torch.nn.ModuleList(modules=(GNN_LayerLin(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.gnn_layers_gate.append(GNN_LayerLin(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_lem = LEMS(2+len(self.eq_variables)+1,self.hidden_features)
        self.lemoutput_mlp = nn.Sequential(
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    Swish())

        self.swish = Swish()

        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
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
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)
        if "D" in self.eq_variables.keys():
            variables = torch.cat((variables, data.D / self.eq_variables["D"]), -1)
        if "r" in self.eq_variables.keys():
            variables = torch.cat((variables, data.r / self.eq_variables["r"]), -1)


        # Encoder and processor (message passing) LEM
        node_input = torch.cat((u, pos_x, variables), -1)
        lem_input = u.unsqueeze(-1).permute((1,0,2))
        lem_input_full = torch.empty(lem_input.size(0),lem_input.size(1),2+len(self.eq_variables)+1).to(device = node_input.device)
        for i in range(lem_input.size(0)):
            lem_input_full[i,:,:] = torch.cat((pos_x,lem_input[i],variables),-1)

        h = self.embedding_lem(lem_input_full)
        h = self.lemoutput_mlp(h)
        
        for i in range(self.hidden_layer):
            tau = torch.sigmoid(self.gnn_layers_gate[i](h, u, pos_x, variables, edge_index, batch))
            tau_m = (torch.ones_like(tau)-tau)
            h = tau_m*h + tau*self.swish(self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch))

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out
