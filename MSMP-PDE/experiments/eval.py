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
from experiments.train_helper import *
from equations.PDEs import *
from experiments.train import *
import matplotlib
from matplotlib import cm
import colorsys
import matplotlib as mpl
from matplotlib import rc
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def plot_2D(pred,true,n = 1,dpi = 600):
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(ncols=2,nrows=2,sharex=True, sharey=True,figsize=(10, 5))

    vmin = -3 #torch.min(allsols)
    vmax = 3 #torch.max(allsols)

    cmap = "viridis" #"jet"
    
    ax2.set_title("Prediction")
    ax1.set_title("Ground Truth")

    im = ax2.imshow(pred[n-1,:,0,:].cpu().transpose(1,0),vmin=vmin, vmax=vmax,cmap=cmap,aspect="auto")
    ax1.imshow(true[n-1,:,0,:].cpu().transpose(1,0),vmin=vmin, vmax=vmax,cmap=cmap,aspect="auto")
    if pred.shape[2] == 2:
        ax4.imshow(pred[n-1,:,1,:].cpu().transpose(1,0),vmin=vmin, vmax=vmax,cmap=cmap,aspect="auto")
        ax3.imshow(true[n-1,:,1,:].cpu().transpose(1,0),vmin=vmin, vmax=vmax,cmap=cmap,aspect="auto")
    ax3.set_xlabel("Timestep")
    ax4.set_xlabel("Timestep")
    ax1.set_ylabel("Grid Point")
    ax3.set_ylabel("Grid Point")

    ax2.yaxis.set_major_locator(MultipleLocator(25))
    ax2.yaxis.set_major_formatter('{x:.0f}')
    # For the minor ticks, use no labels; default NullFormatter.
    ax2.xaxis.set_minor_locator(MultipleLocator(25))
    ax2.xaxis.set_major_locator(MultipleLocator(50))
    ax2.xaxis.set_major_formatter('{x:.0f}')

    #info
    ax2_ref = ax2.twinx()
    ax2_ref.set_ylabel(r"$u_1$",fontsize=15,rotation=0,labelpad=8)
    ax2_ref.set_yticklabels([])
    ax2_ref.set_yticks([])
    ax4_ref = ax4.twinx()
    ax4_ref.set_ylabel(r"$u_2$",fontsize=15,rotation=0,labelpad=8)
    ax4_ref.set_yticklabels([])
    ax4_ref.set_yticks([])

    #colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.93, 0.18, 0.01, 0.7])
    fig.colorbar(im,cax = cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig("plots/plot2d.png",dpi = dpi)

def test_unrolled_losses_plot(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         nr_gt_steps: int,
                         nx_base_resolution: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu",
                         id_to_plot = None) -> torch.Tensor:
    """
    Inefficient function for plotting and stuff
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
    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{euscript}')
    SMALL_SIZE = 8
    #MEDIUM_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    criterion_rel = lambda pred,true : torch.sum(torch.abs(pred-true)/torch.abs(true))
    pred_sol_to_plot = torch.zeros(graph_creator.t_res - (nr_gt_steps-1)*graph_creator.tw, nx_base_resolution)
    true_sol_to_plot = torch.zeros(graph_creator.t_res- (nr_gt_steps-1)*graph_creator.tw, nx_base_resolution)

    #used only in 2d case
    pred_sol_to_plot2 = torch.zeros(graph_creator.t_res- (nr_gt_steps-1)*graph_creator.tw, nx_base_resolution)
    true_sol_to_plot2 = torch.zeros(graph_creator.t_res- (nr_gt_steps-1)*graph_creator.tw, nx_base_resolution)
    
    def scalar_to_rgb(scalar):
        return colorsys.hsv_to_rgb(scalar, 1, 1)
    colors = [scalar_to_rgb(i) for i in np.linspace(0,0.3,graph_creator.t_res - (nr_gt_steps-1)*graph_creator.tw)]
    losses = []
    losses_base = []
    plot_error = []

    losses2 = None
    norms = None

    #store data for single batch plot when not in GNN mode
    base_pred_store = None
    base_true_store = None

    #total number of predicted points
    n_points = graph_creator.t_res - graph_creator.tw*nr_gt_steps
    for batch_id, (u_base, u_super, x, variables) in enumerate(loader):

        #to accomodate when the size of dataset%batch_size != 0
        batch_size = u_super.size(0)

        ###### 
        n = 1
        assert n>=1, "n must be bigger than 1 and smaller or equal to the batch size" 
        ######

        losses_tmp = []
        losses_base_tmp = []

        losses2_tmp = None
        norm_tmp = None
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)
            if f'{model}' == 'GNN':
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                pred = model(graph)
                loss = criterion(pred, graph.y) / nx_base_resolution
                """no = 1
                selec = (graph.edge_index == no)[1,:]
                print(graph.edge_index.shape)
                print(graph.edge_index.permute(1,0)[selec])
                print(x[0,graph.edge_index.permute(1,0)[selec][:,1]]-x[0,no])
                plt.figure(figsize=(10,6),dpi=500)
                plt.grid()
                plt.scatter(x[0,:],torch.ones_like(x[0,:]),marker = ".")
                plt.scatter(x[0,no],1.01,marker=".")
                plt.ylim([0.9,1.2])
                plt.savefig("out.png")
                time.sleep(100)"""

                loss2 = torch.square(pred - graph.y).unflatten(0,(batch_size,graph.y.size(0)//batch_size)).permute(0,2,1)
                norm = torch.square(graph.y).unflatten(0,(batch_size,graph.y.size(0)//batch_size)).permute(0,2,1)

                if batch_id == id_to_plot:
                    #separate 1d and 2d cases
                    if pred.size(-1) == 25:
                        fig, (ax2,ax1) = plt.subplots(2,sharex=True, sharey=True)
                        #store all data
                        base_pred_store = pred.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(pred.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                        base_true_store = ((graph.y.unflatten(0,(batch_size,graph.y.size(0)//batch_size))).unflatten(-1,(graph.y.size(-1)//graph_creator.tw,graph_creator.tw))).permute(0,3,2,1)
                        for i in range(pred.size(-1)):

                            u_in = graph.x[(n-1)*nx_base_resolution:n*nx_base_resolution,i:i+1]
                            u_pred = pred[(n-1)*nx_base_resolution:n*nx_base_resolution,i:i+1]
                            u_true = graph.y[(n-1)*nx_base_resolution:n*nx_base_resolution,i:i+1]

                            min_val, max_val = torch.max(u_in).cpu().numpy(), torch.min(u_in).cpu().numpy()

                            loc_error = criterion(u_pred, u_true)/nx_base_resolution
                            plot_error.append(loc_error)

                            ax1.plot(x[(n-1)],u_in.detach().cpu(),color = colors[i])

                            ax2.plot(x[(n-1)],u_in.detach().cpu(),color = colors[i])

                            ax1.plot(x[(n-1)],u_pred.detach().cpu(),color = colors[pred.size(-1) + i])

                            ax2.plot(x[(n-1)],u_true.detach().cpu(),color = colors[pred.size(-1) + i])

                            pred_sol_to_plot[i,:] = u_in.squeeze().detach().cpu()
                            true_sol_to_plot[i,:] = u_in.squeeze().detach().cpu()
                            pred_sol_to_plot[pred.size(-1)+i,:] = u_pred.squeeze().detach().cpu()
                            true_sol_to_plot[pred.size(-1)+i,:] = u_true.squeeze().detach().cpu()

                    elif pred.size(-1) == 50:
                            #store solutions into array to compute stuff
                            base_pred_store = pred.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(pred.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                            base_true_store = ((graph.y.unflatten(0,(batch_size,graph.y.size(0)//batch_size))).unflatten(-1,(graph.y.size(-1)//graph_creator.tw,graph_creator.tw))).permute(0,3,2,1)

                            u_in1 = graph.x[(n-1)*nx_base_resolution:n*nx_base_resolution,:graph_creator.tw]
                            u_in2 = graph.x[(n-1)*nx_base_resolution:n*nx_base_resolution,graph_creator.tw:]
                            u1_pred = pred[(n-1)*nx_base_resolution:n*nx_base_resolution,:graph_creator.tw]
                            u2_pred = pred[(n-1)*nx_base_resolution:n*nx_base_resolution,graph_creator.tw:]
                            u1_true = graph.y[(n-1)*nx_base_resolution:n*nx_base_resolution,:graph_creator.tw]
                            u2_true = graph.y[(n-1)*nx_base_resolution:n*nx_base_resolution,graph_creator.tw:]

                            loc_error_1 = criterion(u1_pred,u1_true)/nx_base_resolution
                            loc_error_2 = criterion(u2_pred,u2_true)/nx_base_resolution
                            loc_error_rel = compute_relative_error(pred[(n-1)*nx_base_resolution:n*nx_base_resolution,:],graph.y[(n-1)*nx_base_resolution:n*nx_base_resolution,:],time_window=graph_creator.tw)

                            plot_error.append([loc_error_1,loc_error_2])

                            pred_sol_to_plot[0:graph_creator.tw,:] = u_in1.transpose(1,0).detach().cpu()
                            pred_sol_to_plot2[0:graph_creator.tw,:] = u_in2.transpose(1,0).detach().cpu()
                            pred_sol_to_plot[graph_creator.tw:2*graph_creator.tw,:] = u1_pred.transpose(1,0).detach().cpu()
                            pred_sol_to_plot2[graph_creator.tw:2*graph_creator.tw,:] = u2_pred.transpose(1,0).detach().cpu()

                            true_sol_to_plot[0:graph_creator.tw,:] = u_in1.transpose(1,0).detach().cpu()
                            true_sol_to_plot2[0:graph_creator.tw,:] = u_in2.transpose(1,0).detach().cpu()
                            true_sol_to_plot[graph_creator.tw:2*graph_creator.tw,:] = u1_true.transpose(1,0).detach().cpu()
                            true_sol_to_plot2[graph_creator.tw:2*graph_creator.tw,:] = u2_true.transpose(1,0).detach().cpu()


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
                loss2 = torch.square(pred - labels)
                norm = torch.square(labels)
                if len(loss2.shape) == 4:
                    loss2 = loss2.flatten(1,2)
                    norm = norm.flatten(1,2)
                if batch_id == id_to_plot:
                    base_pred_store = pred
                    base_true_store = labels


            losses_tmp.append(loss / batch_size)

            losses2_tmp = loss2
            norm_tmp = norm
            # Unroll trajectory and add losses which are obtained for each unrolling
        
            for step_id,step in enumerate(range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw +1, graph_creator.tw)):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y) / nx_base_resolution
                    loss2 = torch.square(pred - graph.y).unflatten(0,(batch_size,graph.y.size(0)//batch_size)).permute(0,2,1)
                    norm = torch.square(graph.y).unflatten(0,(batch_size,graph.y.size(0)//batch_size)).permute(0,2,1)


                    if batch_id == id_to_plot:
                        #separate 1d and 2d cases
                        if pred.size(-1) == 25:
                            #store all data
                            newpred = pred.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(pred.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                            newtrue = graph.y.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(graph.y.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                            base_pred_store = torch.cat((base_pred_store,newpred),1)
                            base_true_store = torch.cat((base_true_store,newtrue),1)
                            for i in range(pred.size(-1)):

                                u_pred = pred[(n-1)*nx_base_resolution:n*nx_base_resolution,i:i+1]
                                u_true = graph.y[(n-1)*nx_base_resolution:n*nx_base_resolution,i:i+1]

                                loc_error = criterion(u_pred, u_true)/nx_base_resolution
                                plot_error.append(loc_error)

                                maxv, minv = torch.max(u_true).cpu().numpy(), torch.min(u_true).cpu().numpy()
                                max_val = maxv if maxv > max_val else max_val
                                min_val = minv if minv < min_val else min_val


                                ax1.plot(x[(n-1)],u_pred.detach().cpu(),color = colors[step_id*pred.size(-1) + i])

                                ax2.plot(x[(n-1)],u_true.detach().cpu(),color = colors[step_id*pred.size(-1) + i])

                                pred_sol_to_plot[(step_id+2)*pred.size(-1) + i,:] = u_pred.squeeze().detach().cpu()
                                true_sol_to_plot[(step_id+2)*pred.size(-1) + i,:] = u_true.squeeze().detach().cpu()
                        
                        elif pred.size(-1) == 50:
                                #store data
                                newpred = pred.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(pred.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                                newtrue = graph.y.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(graph.y.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                                base_pred_store = torch.cat((base_pred_store,newpred),1)
                                base_true_store = torch.cat((base_true_store,newtrue),1)

                                u1_pred = pred[(n-1)*nx_base_resolution:n*nx_base_resolution,:graph_creator.tw]
                                u2_pred = pred[(n-1)*nx_base_resolution:n*nx_base_resolution,graph_creator.tw:]
                                u1_true = graph.y[(n-1)*nx_base_resolution:n*nx_base_resolution,:graph_creator.tw]
                                u2_true = graph.y[(n-1)*nx_base_resolution:n*nx_base_resolution,graph_creator.tw:]
                                
                                loc_error_1 = criterion(u1_pred,u1_true)/nx_base_resolution
                                loc_error_2 = criterion(u2_pred,u2_true)/nx_base_resolution
                                loc_error_rel = compute_relative_error(pred[(n-1)*nx_base_resolution:n*nx_base_resolution,:],graph.y[(n-1)*nx_base_resolution:n*nx_base_resolution,:],time_window=graph_creator.tw)


                                plot_error.append([loc_error_1,loc_error_2])

                                shift_id = step_id+2
                                pred_sol_to_plot[shift_id*graph_creator.tw:(shift_id+1)*graph_creator.tw,:] = u1_pred.transpose(1,0).detach().cpu()
                                pred_sol_to_plot2[shift_id*graph_creator.tw:(shift_id+1)*graph_creator.tw,:] = u2_pred.transpose(1,0).detach().cpu()
                                true_sol_to_plot[shift_id*graph_creator.tw:(shift_id+1)*graph_creator.tw,:] = u1_true.transpose(1,0).detach().cpu()
                                true_sol_to_plot2[shift_id*graph_creator.tw:(shift_id+1)*graph_creator.tw,:] = u2_true.transpose(1,0).detach().cpu()

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
                    loss2 = torch.square(pred - labels)
                    norm = torch.square(labels)
                    if len(loss2.shape) == 4:
                        loss2 = loss2.flatten(1,2)
                        norm = norm.flatten(1,2)

                    if batch_id == id_to_plot:
                        base_pred_store = torch.cat((base_pred_store,pred),dim = 1)
                        base_true_store = torch.cat((base_true_store,labels),dim = 1)


                losses_tmp.append(loss / batch_size)

                losses2_tmp = torch.cat((losses2_tmp,loss2),1)
                norm_tmp = torch.cat((norm_tmp,norm),1)

            #reset (for the model using the hidden states) the hidden state of LEM for new data
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

        if losses2 is None or norms is None:
            losses2 = losses2_tmp
            norms = norm_tmp
        else:
            losses2 = torch.cat((losses2,losses2_tmp),0)
            norms = torch.cat((norms,norm_tmp),0)

        losses.append(torch.sum(torch.stack(losses_tmp)))

        #print(f"************{torch.sum(torch.stack(losses_tmp))}")
        losses_base.append(torch.sum(torch.stack(losses_base_tmp)))
    losses = torch.stack(losses)
    losses_base = torch.stack(losses_base)
    print("**Non Dimensionless AbsL2**")
    print(f'Unrolled forward losses {torch.mean(losses)}')
    print(f'Unrolled forward base losses {torch.mean(losses_base)}')

    dpi = 400

    #1D plot
    if pred.size(-1) == 25:
        multiplier = 1.2
        ax1.set_ylim(min_val*multiplier,max_val*multiplier)
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax2.yaxis.set_major_formatter('{x:.1f}')
        # For the minor ticks, use no labels; default NullFormatter.
        ax2.xaxis.set_minor_locator(MultipleLocator(1))
        ax2.xaxis.set_major_locator(MultipleLocator(4))
        ax2.xaxis.set_major_formatter('{x:.0f}')

        ax1.set_ylabel(r"$u_{\theta}(x)$")
        ax1.margins(x=0)
        ax1.set_title("Prediction")
        ax2.set_title("Ground Truth")
        ax1.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$u(x)$")
        ax2.margins(x=0)
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.Normalize(vmin=0, vmax=len(colors))
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax = [ax1,ax2])
        cbar.set_label("Timestep",rotation = 270, labelpad=16)
        #fig.suptitle(f"batch_id : {batch_id}, f_id : {n}, abs error : {torch.sum(torch.stack(plot_error))}")
        #fig.suptitle("1D Rollout Plot")
        fig.savefig("plots/plot1d.png",dpi=dpi)
        fig, (ax2,ax1) = plt.subplots(2,sharex=True, sharey=True)
        ax1.imshow(pred_sol_to_plot.transpose(1,0),aspect="auto")
        ax2.imshow(true_sol_to_plot.transpose(1,0),aspect="auto")
        ax1.set_xlabel("Timestep")
        ax1.set_title("Prediction")
        ax2.set_title("Ground Truth")
        ax1.set_ylabel("Grid Point")
        ax2.set_ylabel("Grid Point")
        ax2.yaxis.set_major_locator(MultipleLocator(25))
        ax2.yaxis.set_major_formatter('{x:.0f}')
        # For the minor ticks, use no labels; default NullFormatter.
        ax2.xaxis.set_minor_locator(MultipleLocator(25))
        ax2.xaxis.set_major_locator(MultipleLocator(50))
        ax2.xaxis.set_major_formatter('{x:.0f}')
        #fig.suptitle("2D Rollout Plot")
        fig.savefig("plots/plot2d.png", dpi=dpi)

        #norm on Omega
        p = base_pred_store#[n-1,...][None,...]
        t = base_true_store#[n-1,...][None,...]

        #np.save("plots/KF_n3/true.npy",base_true_store.cpu().numpy())
        #np.save("plots/KF_n3/pred_mppde.npy",base_pred_store.cpu().numpy())

        #[1,n_t,d,n_x]
        _, d = compute_space_L2_norms(torch.square(p-t),torch.square(t))

        fig, ax1 = plt.subplots()
        ax1.set_yscale("log")
        ax1.set_xlabel("Timestep")
        #ax1.set_ylabel(r"$\frac{1}{|\Omega|}\sum_{i } \frac{|u_{\theta}(x_i)-u(x_i)|}{|u(x_i)|} \times 100$")
        ax1.set_ylabel("Relative Error \%")
        fig.suptitle("Rollout Relative Error")
        ax1.set_ylim([1e-2,1])
        x = list(range(25, 225))
        ax1.plot(x,100*d.cpu())
        fig.tight_layout()
        fig.savefig("plots/plot_relerror.png",dpi = dpi)

    #2D system
    if base_pred_store is not None and base_true_store is not None and base_pred_store.size(2) == 2:
        #np.save("plots/RP_n16/pred_FNO.npy",base_pred_store.cpu().numpy())
        #np.save("plots/RP_n16/true.npy",base_true_store.cpu().numpy())
        #2D plot
        plot_2D(base_pred_store,base_true_store,n,dpi)

        #norm on Omega
        p = base_pred_store#[n-1,...][None,...]
        t = base_true_store#[n-1,...][None,...]
        #[1,n_t,d,n_x]
        _, d = compute_space_L2_norms(torch.square(p-t),torch.square(t))

        fig, ax1 = plt.subplots()
        ax1.set_yscale("log")
        ax1.set_xlabel("Timestep")
        #ax1.set_ylabel(r"$\frac{1}{|\Omega|}\sum_{i } \frac{|u_{\theta}(x_i)-u(x_i)|}{|u(x_i)|} \times 100$")
        ax1.set_ylabel("Relative Error \%")
        fig.suptitle("Rollout Relative Error")
        ax1.set_ylim([1e-1,1e2])
        x = list(range(25, 225))
        ax1.plot(x,100*d.cpu())
        fig.tight_layout()
        fig.savefig("plots/plot_relerror.png",dpi = dpi)

        p = base_pred_store[n-1,...][None,...]
        t = base_true_store[n-1,...][None,...]
        num,d = compute_spacetime_L2_norms(torch.square(p-t),torch.square(t))
        print("*Single instance error*")
        print(f"L2 image relative error : {100*d} %")
        print(f"L2 image absolute error : {num}")


    print("**Dimensionless L2 errors**")
    #losses and norms are [B, n_t, n_x] vectors
    #average over space and time to compute the L2(\Omega \times [0,T]) norm
    losses2 = torch.mean(losses2,dim=(1,2))
    norms = torch.mean(norms,dim=(1,2))
    
    #take the sqrt ([batch_size])
    losses2 = torch.sqrt(losses2)
    norms = torch.sqrt(norms)

    #average over batches
    losses2 = torch.mean(losses2)
    norms = torch.mean(norms)
    losses_rel = losses2/norms

    print(f'L2 error {losses2.item()}')
    print(f'L2 relative error {100*losses_rel.item()} %')

    return losses, losses_rel


def long_rollout(model: torch.nn.Module,
                         batch_size: int,
                         nr_gt_steps: int,
                         n_more_rollout: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         device: torch.cuda.device = "cpu") -> torch.Tensor:

    """
    Perform a long rollout of n_more_rollout after the maximum time of the loader
    Args:
        model (torch.nn.Module): neural network PDE solver
        nr_gt_steps (int): number of numerical input timesteps
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        device (torch.cuda.device): device (cpu/gpu)
        n_more_rollout (int): number of additional rollout to perform on top of the standard rollout window
    Returns:
        torch.Tensor: valid/test losses
    """
    #store rollout data in array for plotting
    base_pred_store = None
    base_true_store = None
    for (u_base, u_super, x, variables) in loader:
        with torch.no_grad():
            #first pass
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)
            if f'{model}' == 'GNN':
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                pred = model(graph)
                base_pred_store = pred.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(pred.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                base_true_store = ((graph.y.unflatten(0,(batch_size,graph.y.size(0)//batch_size))).unflatten(-1,(graph.y.size(-1)//graph_creator.tw,graph_creator.tw))).permute(0,3,2,1)
  
            else:
                data, labels = data.to(device), labels.to(device)
                pred = model(data)
                base_pred_store = pred
                base_true_store = labels


            #following passes where numerical solution is still computed
            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                    pred = model(graph)
                    newpred = pred.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(pred.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                    newtrue = graph.y.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(graph.y.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                    base_pred_store = torch.cat((base_pred_store,newpred),1)
                    base_true_store = torch.cat((base_true_store,newtrue),1)
                else:
                    labels = labels.to(device)
                    pred = model(pred)
                    base_pred_store = torch.cat((base_pred_store,pred),dim = 1)
                    base_true_store = torch.cat((base_true_store,labels),dim = 1)

            #rollout outside of training/validation/testing window
            for step in range(n_more_rollout):
                if f'{model}' == 'GNN':
                    #use the previous prediction
                    graph.x = pred
                    pred = model(graph)
                    newpred = pred.unflatten(0,(batch_size,graph.y.size(0)//batch_size)).unflatten(-1,(pred.size(-1)//graph_creator.tw,graph_creator.tw)).permute(0,3,2,1)
                    base_pred_store = torch.cat((base_pred_store,newpred),1)
                    base_true_store = torch.cat((base_true_store,torch.zeros_like(newpred)),dim = 1)
                else:
                    pred = model(pred)
                    base_pred_store = torch.cat((base_pred_store,pred),dim = 1)
                    base_true_store = torch.cat((base_true_store,torch.zeros_like(pred)),dim = 1)

    return base_pred_store, base_true_store


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

    
    """total_dataset = torch.utils.data.ConcatDataset([train_dataset,valid_dataset,test_dataset])
    test_idx=
    my_subset = torch.utils.data.Subset(total_dataset, test_idx)
    test_loader = DataLoader(my_subset,
                             batch_size=args.batch_size,
                             shuffle=False)"""

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
        elif (args.experiment == 'RP') or args.experiment == 'MSWG' or args.experiment == 'MSWG3' or args.experiment == 'RPU':
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

    criterion = torch.nn.MSELoss(reduction="sum")    

    if args.model_to_test is not None:
        print(f"Eval Mode for model : {args.model_to_test}")
        model.load_state_dict(torch.load(args.model_to_test))
        model.eval()
        steps = [t for t in range(graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
        # next we test the unrolled losses
        print("******** Norm func ********")
        losses2, losses_rel = compute_L2_norms(model=model,
                                    batch_size=args.batch_size,
                                    nr_gt_steps=args.nr_gt_steps,
                                    loader=test_loader,
                                    graph_creator=graph_creator,
                                    device=device)
        print("******** Plot func ********")
        losses, losses_rel = test_unrolled_losses_plot(model=model,
                                    steps=steps,
                                    batch_size=args.batch_size,
                                    nr_gt_steps=args.nr_gt_steps,
                                    nx_base_resolution=args.base_resolution[1],
                                    loader=test_loader,
                                    graph_creator=graph_creator,
                                    criterion=criterion,
                                    device=device,
                                    id_to_plot=0)
        print("******** Long Rollout ********")
        """p,t = long_rollout(model=model,
                                    batch_size=args.batch_size,
                                    nr_gt_steps=args.nr_gt_steps,
                                    n_more_rollout=45,
                                    loader=test_loader,
                                    graph_creator=graph_creator,
                                    device=device)
        print(p.shape,t.shape)
        plot_2D(p,t,1)"""
        print("******** Train func ********")
        losses2 = test_unrolled_losses(model=model,
                                    steps=steps,
                                    batch_size=args.batch_size,
                                    nr_gt_steps=args.nr_gt_steps,
                                    nx_base_resolution=args.base_resolution[1],
                                    loader=test_loader,
                                    graph_creator=graph_creator,
                                    criterion=criterion,
                                    device=device)

    else:
        assert("model to test not given")


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
                        default=2, help="Number of steps done by numerical solver")
    parser.add_argument('--n_graph_layers', type=int,
                        default=6, help="Number of steps done by numerical solver")

    # Misc
    parser.add_argument('--print_interval', type=int, default=20,
            help='Interval between print statements')
    parser.add_argument('--log', type=eval, default=False,
            help='pip the output to log file')
    parser.add_argument('--model_to_test', type=str, default=None,
            help='model file path to evaluate')

    args = parser.parse_args()
    main(args)
    print(f"Elapsed Time : {time.time() - ts}")
