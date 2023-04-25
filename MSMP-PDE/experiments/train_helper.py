import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from common.utils import HDF5Dataset, GraphCreator
from experiments.models_gnn2D import unflatten_u
from equations.PDEs import *
from experiments.models_gnn import MP_PDE_SolverLEMLinGatedSave

def reset_state_bool(model):
    inst = isinstance(model, MP_PDE_SolverLEMLinGatedSave)
    has = hasattr(model,"save_state")
    return inst or (has and model.save_state is not None and model.save_state)

#####DEPRECATED#####
def compute_relative_error(pred : torch.Tensor, true : torch.Tensor, batch_size : int = 1, time_window : float = 1):
    assert true.shape == pred.shape,  "The two input tensors should have the same shape"

    #need to accumulate also on the dimension --> [batch_size*n_x,dim,time_window]
    if pred.size(1) != time_window:
        pred = unflatten_u(pred,time_window)
        true = unflatten_u(true,time_window)

    #[batch_size*n_x,dim,time_window] or [batch_size*n_x,time_window]
    delta = torch.abs(pred-true)
    true = torch.abs(true)
    delta = delta.unflatten(0,(batch_size,delta.size(0)//batch_size))
    true = true.unflatten(0,(batch_size,true.size(0)//batch_size))

    #[batch_size,n_x,dim,time_window] or #[batch_size,n_x,time_window]

    if pred.size(1) != time_window:
        #--> [batch_size,n_x*dim,time_window]
        delta = delta.flatten(1,2)
        true = true.flatten(1,2)

    delta = torch.sum(delta,1)
    true = torch.sum(true,1)

    #[batch_size,time_window]
    rel_error = delta/true

    #average over batch size and sum over times, the time averaging is done later according to
    #the true time window
    return torch.sum(rel_error)/(batch_size*time_window)
    
#####DEPRECATED#####
def compute_relative_error_2(pred : torch.Tensor, true : torch.Tensor, batch_size : int = 1, time_window : float = 1):
    #Only compatible with 1D for now
    assert true.shape == pred.shape,  "The two input tensors should have the same shape"

    #[batch_size,time_window,n_x]
    delta = torch.abs(pred-true)
    true = torch.abs(true)

    delta = torch.sum(delta,2)
    true = torch.sum(true,2)

    #[batch_size,time_window]
    rel_error = delta/true

    #average over batch size and sum over times, the time averaging is done later according to
    #the true time window
    return torch.sum(rel_error)/(batch_size*time_window)

def training_loop(model: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator,
                  criterion: torch.nn.modules.loss,
                  device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory
    Args:
        model (torch.nn.Module): neural network PDE solver
        unrolling (list): list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """

    losses = []
    for (u_base, u_super, x, variables) in loader:
        optimizer.zero_grad()
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                  graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u_super, random_steps)
        if f'{model}' == 'GNN':
            graph = graph_creator.create_graph(data, labels, x, variables, random_steps).to(device)
        else:
            data, labels = data.to(device), labels.to(device)

        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_graphs):
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u_super, random_steps)
                if f'{model}' == 'GNN':
                    pred = model(graph)
                    graph = graph_creator.create_next_graph(graph, pred, labels, random_steps).to(device)
                else:
                    #check if the model takes the equation parameters as input or if need the grid (FNO)
                    if hasattr(model,'eq_variables'):
                        if hasattr(model,'interp') and model.interp :
                            data = model(data,variables,x)
                        else :
                            data = model(data,variables)
                    else :
                        data = model(data)
                    labels = labels.to(device)

        if f'{model}' == 'GNN':
            pred = model(graph)
            loss = criterion(pred, graph.y)
        else:
            #check if the model takes the equation parameters as input or if need the grid (FNO)
            if hasattr(model,'eq_variables'):
                if hasattr(model,'interp') and model.interp :
                    pred = model(data,variables,x)
                else :
                    pred = model(data,variables)
            else :
                pred = model(data)
            loss = criterion(pred, labels)

        loss = torch.sqrt(loss)
        loss.backward()
        losses.append(loss.detach() / batch_size)
        optimizer.step()

        #reset the hidden state of LEM for new data
        if reset_state_bool(model):
            model.embedding_lem.reset_states()

    losses = torch.stack(losses)
    return losses

def test_timestep_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> None:
    """
    Loss for one neural network forward pass at certain timepoints on the validation/test datasets
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        batch_size (int): batch size
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """

    for step in steps:

        if (step != graph_creator.tw and step % graph_creator.tw != 0):
            continue

        losses = []
        for (u_base, u_super, x, variables) in loader:
            with torch.no_grad():
                same_steps = [step]*batch_size
                data, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y)
                else:
                    data, labels = data.to(device), labels.to(device)
                    #check if the model takes the equation parameters as input or if need the grid (FNO)
                    if hasattr(model,'eq_variables'):
                        if hasattr(model,'interp') and model.interp :
                            pred = model(data,variables,x)
                        else :
                            pred = model(data,variables)
                    else :
                        pred = model(data)
                    loss = criterion(pred, labels)
                losses.append(loss / batch_size)

            #reset the hidden state of LEM for new data
            if reset_state_bool(model):
                model.embedding_lem.reset_states()

        losses = torch.stack(losses)
        print(f'Step {step}, mean loss {torch.mean(losses)}')

def test_unrolled_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         nr_gt_steps: int,
                         nx_base_resolution: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> torch.Tensor:
    """
    Loss for full trajectory unrolling
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        nr_gt_steps (int): number of numerical input timesteps
        nx_base_resolution (int): spatial resolution of numerical baseline
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = []
    losses_base = []
    for (u_base, u_super, x, variables) in loader:
        losses_tmp = []
        losses_base_tmp = []
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)
            if f'{model}' == 'GNN':
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                pred = model(graph)
                loss = criterion(pred, graph.y) / nx_base_resolution
            else:
                data, labels = data.to(device), labels.to(device)
                #check if the model takes the equation parameters as input or if need the grid (FNO)
                if hasattr(model,'eq_variables'):
                    if hasattr(model,'interp') and model.interp :
                        pred = model(data,variables,x)
                    else :
                        pred = model(data,variables)
                else :
                    pred = model(data)
                loss = criterion(pred, labels) / nx_base_resolution

            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y) / nx_base_resolution
                else:
                    labels = labels.to(device)
                    #check if the model takes the equation parameters as input or if need the grid (FNO)
                    if hasattr(model,'eq_variables'):
                        if hasattr(model,'interp') and model.interp :
                            pred = model(pred,variables,x)
                        else :
                            pred = model(pred,variables)
                    else :
                        pred = model(data)
                    loss = criterion(pred, labels) / nx_base_resolution
                losses_tmp.append(loss / batch_size)

            #reset the hidden state of LEM for new data
            if reset_state_bool(model):
                model.embedding_lem.reset_states()

            # Losses for numerical baseline
            for step in range(graph_creator.tw * nr_gt_steps, graph_creator.t_res - graph_creator.tw + 1,
                              graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels_super = graph_creator.create_data(u_super, same_steps)
                _, labels_base = graph_creator.create_data(u_base, same_steps)
                loss_base = criterion(labels_super, labels_base) / nx_base_resolution
                losses_base_tmp.append(loss_base / batch_size)

        losses.append(torch.sum(torch.stack(losses_tmp)))
        losses_base.append(torch.sum(torch.stack(losses_base_tmp)))

    losses = torch.stack(losses)
    losses_base = torch.stack(losses_base)
    print(f'Unrolled forward losses {torch.mean(losses)}')
    print(f'Unrolled forward base losses {torch.mean(losses_base)}')

    return losses

@torch.jit.script
def compute_spacetime_L2_norms(losses : torch.Tensor, norms : torch.Tensor):
    """
    Compute the norm on L2(\Omega \times [0,T]), absolute and relative, input of size [B, n_t, d, n_x]
    Args:
        losses (torch.Tensor): torch.square(pred-true)
        norms (torch.Tensor): torch.square(pred)
    Returns:
        torch.Tensor: valid/test loss and relative loss scalar
    """
    assert losses.shape == norms.shape, "loss and norms do not have the same shape"
    # sum over d -->  [B, n_t, n_x] (norm on R^2)
    losses = torch.sum(losses, dim=2)
    norms = torch.sum(norms,dim=2)

    #average over space and time --> [B] (norm on L2(\Omega \times [0,T])
    losses = torch.mean(losses, dim = (1,2))
    norms = torch.mean(norms, dim=(1,2))

    #take the sqrt ([B])
    losses = torch.sqrt(losses)
    norms = torch.sqrt(norms)

    #average over testing samples
    losses = torch.mean(losses)
    norms = torch.mean(norms)

    losses_rel = losses/norms

    #scalar
    return losses, losses_rel

@torch.jit.script
def compute_space_L2_norms(losses : torch.Tensor, norms : torch.Tensor):
    """
    Compute the norm on L2(\Omega), absolute and relative, input of size [B, n_t, d, n_x]
    Args:
        losses (torch.Tensor): torch.square(pred-true)
        norms (torch.Tensor): torch.square(pred)
    Returns:
        torch.Tensor: valid/test losses and relative losses array
    """
    assert losses.shape == norms.shape, "loss and norms do not have the same shape"

    # sum over d -->  [B, n_t, n_x] (norm on R^2)
    losses = torch.sum(losses, dim=2)
    norms = torch.sum(norms,dim=2)

    #average over space and time --> [B, n_t] (norm on L2(\Omega)) 
    losses = torch.mean(losses, dim=2)
    norms = torch.mean(norms, dim=2)

    #take the sqrt ([B])
    losses = torch.sqrt(losses)
    norms = torch.sqrt(norms)

    #average over testing samples
    losses = torch.mean(losses,dim = 0)
    norms = torch.mean(norms,dim = 0)
    losses_rel = losses/norms

    #[n_t]    
    return losses, losses_rel

def compute_L2_norms(model: torch.nn.Module,
                         batch_size: int,
                         nr_gt_steps: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         device: torch.cuda.device = "cpu") -> torch.Tensor:
    """
    L2 absolute and relative error for full trajectory unrolling, we report this error in the paper.
    Args:
        model (torch.nn.Module): neural network PDE solver
        nr_gt_steps (int): number of numerical input timesteps
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = None
    norms = None
    for (u_base, u_super, x, variables) in loader:
        batch_size = u_super.size(0)

        #check if we are dealing with a system or scalar equation
        if len(u_super.shape) == 4:
            d = u_super.size(2)
        else:
            d = 1

        losses_tmp = None
        norm_tmp = None
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)
            if f'{model}' == 'GNN':
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                pred = model(graph)#[B*n_x, d*tw]

                #separate batch_size from n_x and d from tw --> #[B, tw, d, n_x]
                loss = torch.square(pred - graph.y).unflatten(0,(batch_size,graph.y.size(0)//batch_size)).permute(0,2,1).unflatten(1,(graph_creator.tw,d))
                norm = torch.square(graph.y).unflatten(0,(batch_size,graph.y.size(0)//batch_size)).permute(0,2,1).unflatten(1,(graph_creator.tw,d))
            else:
                data, labels = data.to(device), labels.to(device)
                #check if the model takes the equation parameters as input or if need the grid (FNO)
                if hasattr(model,'eq_variables'):
                    if hasattr(model,'interp') and model.interp :
                        pred = model(data,variables,x)
                    else :
                        pred = model(data,variables)
                else :
                    pred = model(data)
                loss = torch.square(pred - labels)
                norm = torch.square(labels)
                if d == 1:
                    loss = loss.unflatten(1,(graph_creator.tw,d))
                    norm = norm.unflatten(1,(graph_creator.tw,d))

            losses_tmp = loss
            norm_tmp = norm

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                    pred = model(graph) #[B*n_x, d*tw]

                    #separate batch_size from n_x --> #[B, d*tw, n_x]
                    loss = torch.square(pred - graph.y).unflatten(0,(batch_size,graph.y.size(0)//batch_size)).permute(0,2,1).unflatten(1,(graph_creator.tw,d))
                    norm = torch.square(graph.y).unflatten(0,(batch_size,graph.y.size(0)//batch_size)).permute(0,2,1).unflatten(1,(graph_creator.tw,d))
                else:
                    labels = labels.to(device)
                    #check if the model takes the equation parameters as input or if need the grid (FNO)
                    if hasattr(model,'eq_variables'):
                        if hasattr(model,'interp') and model.interp :
                            pred = model(pred,variables,x)
                        else :
                            pred = model(pred,variables)#[B, tw, d, n_x]
                    else :
                        pred = model(data)
                    loss = torch.square(pred - labels)
                    norm = torch.square(labels)
                    if d == 1:
                        loss = loss.unflatten(1,(graph_creator.tw,d))
                        norm = norm.unflatten(1,(graph_creator.tw,d))
 
                #aggregate over time
                losses_tmp = torch.cat((losses_tmp,loss),1)
                norm_tmp = torch.cat((norm_tmp,norm),1)

            #reset the hidden state of LEM for new data
            if reset_state_bool(model):
                model.embedding_lem.reset_states()

        if losses is None or norms is None:
            losses = losses_tmp
            norms = norm_tmp
        else:
            #aggregate over batches
            losses = torch.cat((losses,losses_tmp),0)
            norms = torch.cat((norms,norm_tmp),0)

    #losses and norms are [B, n_t, d, n_x] vectors
    #average over space and time to compute the L2(\Omega \times [0,T]) norm
    losses,losses_rel = compute_spacetime_L2_norms(losses,norms)

    print(f'L2 error {losses.item()}')
    print(f'L2 relative error {100*losses_rel.item()} %')

    return losses.item(), losses_rel.item()
