import argparse
import os
import copy
import sys
import time
from datetime import datetime
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.utils import HDF5Dataset, GraphCreator
from experiments.models_gnn import MP_PDE_Solver, MP_PDE_SolverGated, MP_PDE_SolverLEMLin, MP_PDE_SolverLEMLinGated, MP_PDE_SolverLSTMLinGated, MP_PDE_SolverLSTMLin, MSSMP_PDE_Solver, MP_PDE_SolverLEMLinGatedSave, MP_PDE_SolverLEMLinGatedGLU
from experiments.models_gnn2D import MP_PDE_Solver2D, MP_PDE_Solver2DGated, MP_PDE_Solver2DLEMLinGated,MP_PDE_Solver2DLEMLin, MP_PDE_Solver2DLSTMLin, MP_PDE_Solver2DLSTMLinGated, G_PDE_Solver2DLEMLinGated, MP_PDE_Solver2DLEMLinG2, MP_PDE_Solver2DLEMLinGatedGLU
from experiments.models_fno import FNO1d, FNO1dParam, VNO1d
from experiments.models_fno2D import FNO2d, FNO2dParam, FNO2dParamUn
from experiments.models_cnn2D import BaseCNN2D
from experiments.models_cnn import BaseCNN
from experiments.train_helper import *
from equations.PDEs import *

def getModel(graph_creator, device, args, pde, eq_variables):
    print("***Model***")
    print(f"learning_rate : {args.lr}")
    print(f"learning_rate_decay : {args.lr_decay}")
    print(f"num_epochs : {args.num_epochs}")
    print(f"time_window : {args.time_window}")      
    print(f"batch_size : {args.batch_size}")      
    print(f"unrolling : {args.unrolling}")      
    print(f"nr_gt_steps : {args.nr_gt_steps}")      

    if args.model == 'MP-PDE':
        model = MP_PDE_Solver(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'BaseCNN':
        model = BaseCNN(pde=pde,
                        time_window=args.time_window).to(device)
    elif args.model == 'Gated':
        model = MP_PDE_SolverGated(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'LEM':
        model = MP_PDE_SolverLEMLin(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'MSMP-PDE':
        model = MP_PDE_SolverLEMLinGated(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'MSSMP-PDE':
        model = MSSMP_PDE_Solver(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'MSGMP-PDE':
        model = MP_PDE_SolverLEMLinGatedGLU(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'SaveMSMP-PDE':
        model = MP_PDE_SolverLEMLinGatedSave(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'LSTMGated':
        model = MP_PDE_SolverLSTMLinGated(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'LSTM':
        model = MP_PDE_SolverLSTMLin(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'FNO':
        modes = 16
        width = 64
        print(f"fno_modes : {modes}")      
        print(f"fno_width : {width}")      
        domain = [0.0,pde.L]
        model = FNO1d(pde=pde,
                        modes=modes, width=width, input_size=args.time_window, output_size=args.time_window,domain = domain).to(device) 
    elif args.model == 'VNO':
        modes = 16
        width = 64
        print(f"fno_modes : {modes}")      
        print(f"fno_width : {width}")      
        domain = [0.0,pde.L]
        model = VNO1d(pde=pde,
                        modes=modes, width=width, input_size=args.time_window, output_size=args.time_window,domain = domain, batch_size = args.batch_size).to(device) 
    elif args.model == 'FNOP':
        modes = 16
        width = 64
        print(f"fno_modes : {modes}")      
        print(f"fno_width : {width}")      
        domain = [0.0,pde.L]
        model = FNO1dParam(pde=pde,
                        modes=modes, width=width, input_size=args.time_window, output_size=args.time_window,domain = domain, eq_variables=eq_variables).to(device) 
    elif args.model == 'MP-PDE2D':
        model = MP_PDE_Solver2D(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'Gated2D':
        model = MP_PDE_Solver2DGated(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'MSMP-PDE2D':
        model = MP_PDE_Solver2DLEMLinGated(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device) 
    elif args.model == 'MSGMP-PDE2D':
        model = MP_PDE_Solver2DLEMLinGatedGLU(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device) 
    elif args.model == 'SaveMSMP-PDE2D':
        model = MP_PDE_Solver2DLEMLinGated(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers,
                              save_state=True).to(device)
    elif args.model == 'MSG2-PDE2D':
        model = MP_PDE_Solver2DLEMLinG2(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device) 
    elif args.model == 'BaseCNN2D':
        model = BaseCNN2D(pde=pde,
                        time_window=args.time_window).to(device)
    elif args.model == 'LSTMGated2D':
        model = MP_PDE_Solver2DLSTMLinGated(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device) 
    elif args.model == 'LEM2D':
        model = MP_PDE_Solver2DLEMLin(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device) 
    elif args.model == 'GLEMGated2D':
        model = G_PDE_Solver2DLEMLinGated(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'LSTM2D':
        model = MP_PDE_Solver2DLSTMLin(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              hidden_layer=args.n_graph_layers).to(device)
    elif args.model == 'FNO2D':
        modes = 16
        width = 128
        print(f"fno_modes : {modes}")      
        print(f"fno_width : {width}")  
        domain = [0.0,pde.L]
        model = FNO2d(pde=pde,
                        modes=modes, width=width, input_size=args.time_window, output_size=args.time_window,domain = domain).to(device) 
    elif args.model == 'FNO2DP':
        modes = 16
        width = 128
        print(f"fno_modes : {modes}")      
        print(f"fno_width : {width}")  
        domain = [0.0,pde.L]
        model = FNO2dParam(pde=pde,
                        modes=modes, width=width, input_size=args.time_window, output_size=args.time_window,domain = domain, eq_variables=eq_variables).to(device)
    elif args.model == 'FNO2DPU':
        modes = 16
        width = 128
        print(f"fno_modes : {modes}")      
        print(f"fno_width : {width}")  
        domain = [0.0,pde.L]
        model = FNO2dParamUn(pde=pde,
                        modes=modes, width=width, input_size=args.time_window, output_size=args.time_window,domain = domain, eq_variables=eq_variables).to(device) 
    else:
        raise Exception("Wrong model specified")


    if f"{model}" == "GNN":
        print(f"n_neighbors : {args.neighbors}")  
        print(f"n_graph_layers : {args.n_graph_layers}")  
    
    print("**********")

    return model

def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f'experiments/log'):
        os.mkdir(f'experiments/log')
    if not os.path.exists(f'models'):
        os.mkdir(f'models')

def train(args: argparse,
          pde: PDE,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          loader: DataLoader,
          graph_creator: GraphCreator,
          criterion: torch.nn.modules.loss,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    model.train()

    # Sample number of unrolling steps during training (pushforward trick)
    # Default is to unroll zero steps in the first epoch and then increase the max amount of unrolling steps per additional epoch.
    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    # Loop over every epoch as often as the number of timesteps in one trajectory.
    # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is covered.
    for i in range(graph_creator.t_res):
        losses = training_loop(model, unrolling, args.batch_size, optimizer, loader, graph_creator, criterion, device)
        if(i % args.print_interval == 0):
            print(f'Training Loss (progress: {i / graph_creator.t_res:.2f}): {torch.mean(losses)}')

def test(args: argparse,
         pde: PDE,
         model: torch.nn.Module,
         loader: DataLoader,
         graph_creator: GraphCreator,
         criterion: torch.nn.modules.loss,
         device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    Test routine
    Both step wise and unrolled forward losses are computed
    and compared against low resolution solvers
    step wise = loss for one neural network forward pass at certain timepoints
    unrolled forward loss = unrolling of the whole trajectory
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled forward loss
    """
    model.eval()

   # first we check the losses for different timesteps (one forward prediction array!)
    steps = [t for t in range(graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
    losses = test_timestep_losses(model=model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)

    # next we test the unrolled losses
    losses = test_unrolled_losses(model=model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  nr_gt_steps=args.nr_gt_steps,
                                  nx_base_resolution=args.base_resolution[1],
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)

    return torch.mean(losses)


def main(args: argparse):

    device = args.device
    check_directory()

    base_resolution = args.base_resolution
    super_resolution = args.super_resolution

    # Check for experiments and if resolution is available
    if args.experiment == 'E1' or args.experiment == 'E2' or args.experiment == 'E3':
        pde = CE(device=device)
        assert(base_resolution[0] == 250)
        assert(base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40)
    elif args.experiment == 'WE1' or args.experiment == 'WE2' or args.experiment == 'WE3':
        pde = WE(device=device)
        assert (base_resolution[0] == 250)
        assert (base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40 or base_resolution[1] == 20)
        if args.model != 'GNN':
            raise Exception("Only MP-PDE Solver is implemented for irregular grids so far.")
    elif args.experiment == 'KF':
        pde = KF(device=device)
        assert(base_resolution[0] == 250)
        assert(base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40)
    elif args.experiment == 'KS':
        pde = KS()
        assert(base_resolution[0] == 250 or base_resolution[0] == 500)
        assert(base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40)
    elif args.experiment == 'RP' or args.experiment == 'MSWG' or args.experiment == 'MSWG3' or args.experiment == 'RPU':
        pde = AD(device=device)
        assert(base_resolution[0] == 250 or base_resolution[0] == 500)
        assert(base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40)
        if args.experiment == 'RPU':
            pde.untructured_grid = True
    else:
        raise Exception("Wrong experiment")

    # Load datasets
    train_string = f'data/{pde}_train_{args.experiment}.h5'
    valid_string = f'data/{pde}_valid_{args.experiment}.h5'
    test_string = f'data/{pde}_test_{args.experiment}.h5'
    try:
        train_dataset = HDF5Dataset(train_string, pde=pde, mode='train', base_resolution=base_resolution, super_resolution=super_resolution)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

        valid_dataset = HDF5Dataset(valid_string, pde=pde, mode='valid', base_resolution=base_resolution, super_resolution=super_resolution)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False)

        test_dataset = HDF5Dataset(test_string, pde=pde, mode='test', base_resolution=base_resolution, super_resolution=super_resolution)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    except:
        raise Exception("Datasets could not be loaded properly")

    # Equation specific parameters
    pde.tmin = train_dataset.tmin
    pde.tmax = train_dataset.tmax
    pde.grid_size = base_resolution
    pde.dt = train_dataset.dt

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'

    if(args.log):
        logfile = f'experiments/log/{args.model}_{pde}_{args.experiment}_xresolution{args.base_resolution[1]}-{args.super_resolution[1]}_n{args.neighbors}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}.csv'
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')

    save_path = f'models/{args.model}_{pde}_{args.experiment}_xresolution{args.base_resolution[1]}-{args.super_resolution[1]}_n{args.neighbors}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}.pt'
    print(f'Training on dataset {train_string}')
    print(device)
    print(save_path)

    # Equation specific input variables
    eq_variables = {}
    if not args.parameter_ablation:
        if args.experiment == 'E2':
            print(f'Beta parameter added to the GNN solver')
            eq_variables['beta'] = 0.2
        elif args.experiment == 'E3':
            print(f'Alpha, beta, and gamma parameter added to the GNN solver')
            eq_variables['alpha'] = 3.
            eq_variables['beta'] = 0.4
            eq_variables['gamma'] = 1.
        elif (args.experiment == 'WE3'):
            print('Boundary parameters added to the GNN solver')
            eq_variables['bc_left'] = 1
            eq_variables['bc_right'] = 1
        elif (args.experiment == 'KF'):
            print('Diffusion and reaction parameters added to the solver')
            eq_variables['D'] = 1e-4
            eq_variables['r'] = 1.0
        elif args.experiment == 'RP' or args.experiment == 'MSWG' or args.experiment == 'MSWG3' or args.experiment == 'RPU':
            print('Two speeds added to the solver')
            eq_variables['a'] = 1.
            eq_variables['b'] = 1.

    graph_creator = GraphCreator(pde=pde,
                                 neighbors=args.neighbors,
                                 time_window=args.time_window,
                                 t_resolution=args.base_resolution[0],
                                 x_resolution=args.base_resolution[1]).to(device)

    model = getModel(graph_creator,device,args,pde,eq_variables)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of parameters: {params}')

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.unrolling, 5, 10, 15], gamma=args.lr_decay)

    # Training loop
    min_val_loss = 10e30
    test_loss = 10e30
    criterion = torch.nn.MSELoss(reduction="sum")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        train(args, pde, epoch, model, optimizer, train_loader, graph_creator, criterion, device=device)
        print("Evaluation on validation dataset:")
        val_loss = test(args, pde, model, valid_loader, graph_creator, criterion, device=device)
        if(val_loss < min_val_loss):
            print("Evaluation on test dataset:")
            test_loss = test(args, pde, model, test_loader, graph_creator, criterion, device=device)
            print("**Dimensionless L2 errors**")
            print("*Valid*")
            valid_error_L2, valid_error_rel_L2 = compute_L2_norms(model, args.batch_size, args.nr_gt_steps, valid_loader, graph_creator, device)
            print("*Test*")
            test_error_L2, test_error_rel_L2 = compute_L2_norms(model, args.batch_size, args.nr_gt_steps, test_loader, graph_creator, device)

            # Save model
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at {save_path}\n")
            min_val_loss = val_loss

        scheduler.step()

    print(f"Min Val loss: {min_val_loss}")
    print(f"Test loss: {test_loss}")
    print("\n")

    print("**Dimensionless L2 errors**")
    print(f"Min Val L2 Error: {valid_error_L2}")
    print(f"Min Relative Val L2 Error: {100*valid_error_rel_L2} %")
    print(f"Test L2 Error: {test_error_L2}")
    print(f"Relative Test L2 Error: {100*test_error_rel_L2} %")


if __name__ == "__main__":
    ts = time.time()
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    # PDE
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--experiment', type=str, default='',
                        help='Experiment for PDE solver should be trained: [E1, E2, E3, WE1, WE2, WE3]')

    # Model
    parser.add_argument('--model', type=str, default='GNN',
                        help='Model used as PDE solver: [GNN, BaseCNN]')

    # Model parameters
    parser.add_argument('--batch_size', type=int, default=16,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
            help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Learning rate')
    parser.add_argument('--lr_decay', type=float,
                        default=0.4, help='multistep lr decay')
    parser.add_argument('--parameter_ablation', type=eval, default=False,
                        help='Flag for ablating MP-PDE solver without equation specific parameters')

    # Base resolution and super resolution
    parser.add_argument('--base_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 100], help="PDE base resolution on which network is applied")
    parser.add_argument('--super_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 200], help="PDE super resolution for calculating training and validation loss")
    parser.add_argument('--neighbors', type=int,
                        default=3, help="Neighbors to be considered in GNN solver")
    parser.add_argument('--time_window', type=int,
                        default=25, help="Time steps to be considered in GNN solver")
    parser.add_argument('--unrolling', type=int,
                        default=1, help="Unrolling which proceeds with each epoch")
    parser.add_argument('--nr_gt_steps', type=int,
                        default=2, help="Number of steps done by numerical solver before feeding into neural network," \
                        "ex: (nr_gt_steps - 1)*(time_window):nr_gt_steps*(time_window) is used as first input for" \
                        "computation of the unrolled test error.")
    parser.add_argument('--n_graph_layers', type=int,
                        default=6, help="Number of steps done by numerical solver")

    # Misc
    parser.add_argument('--print_interval', type=int, default=20,
            help='Interval between print statements')
    parser.add_argument('--log', type=eval, default=False,
            help='pip the output to log file')

    args = parser.parse_args()
    main(args)
    print(f"Elapsed Time : {time.time() - ts}")
