'''
We adapt the codes in paper "Robust pricing and hedging via neural SDEs" (https://arxiv.org/abs/2007.04154) by Gierjatowicz et. al. to conduct our own testing. 
This script in particular implements the local volatility (LV) model calibration to vanilla option prices.
'''


import sys
import os

sys.path.append(os.path.dirname('__file__'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import time
from random import randrange
import copy
import argparse
import random

from applications.nsde_calibration.networks import *

class Net_LV(nn.Module):
    """
    Calibration of LV model: dS_t = S_t*r*dt + L(t,S_t,theta)dW_t to vanilla prices at different maturities
    """

    def __init__(self, dim, timegrid, strikes_call,  n_layers, vNetWidth, device, rate, maturities, n_maturities):
        
        super(Net_LV, self).__init__()
        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.maturities = maturities
        self.rate = rate
        
        
        # Leverage function
        self.diffusion = Net_timegrid(dim=dim+1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, n_maturities=n_maturities, activation_output="softplus")
        
        # Control Variates
        self.control_variate_vanilla = Net_timegrid(dim=dim+1, nOut=len(strikes_call)*n_maturities, n_layers=3, vNetWidth=30, n_maturities=n_maturities)
        self.control_variate_exotics = Net_timegrid(dim=dim*len(self.timegrid)+1, nOut=1, n_layers = 3, vNetWidth = 20, n_maturities=n_maturities)
        


    def forward(self, S0, z, MC_samples, ind_T, period_length=30):
        """
        Computes stock paths and variance paths under a Local Volatility (LV) model.

        Parameters
        ----------
        S0 : Tensor
            Initial stock price (batch size, 1)
        z : Tensor
            Brownian motion samples of shape (MC_samples, timesteps)
        MC_samples : int
            Number of Monte Carlo samples
        ind_T : int
            Index of the final time step to simulate
        period_length : int, optional
            Period length for control variates (default is 30)

        Returns
        -------
        path : Tensor
            Simulated stock price paths (MC_samples, timegrid steps)
        var_path : Tensor
            Variance paths (MC_samples, timegrid steps)
        diffusion : Tensor
            Final diffusion term \( \sigma(S, t) \)
        price_vanilla_cv : Tensor
            Vanilla option prices using control variates
        var_price_vanilla_cv : Tensor
            Variance of vanilla option prices
        exotic_option_price : Tensor
            Exotic option prices
        exotic_option_price.mean() : float
            Mean exotic option price
        exotic_option_price.var() : float
            Variance of exotic option price
        error : Tensor
            Pricing error based on control variates
        """

        ones = torch.ones(MC_samples, 1, device=self.device)
        path = torch.zeros(MC_samples, len(self.timegrid), device=self.device)  # Stock paths
        var_path = torch.zeros(MC_samples, len(self.timegrid), device=self.device)  # Variance paths

        S_old = ones * S0
        path[:, 0] = S_old.squeeze(1)

        # Initial variance from LV model (computed as σ²)
        diffusion = self.diffusion.forward_idx(0, torch.cat([torch.zeros_like(S_old), S_old], 1))
        var_path[:, 0] = diffusion.detach().squeeze(1) ** 2  # Store variance as σ²(S,t)

        running_max = S_old  # Track max stock price for exotic options

        cv_vanilla = torch.zeros(S_old.shape[0], len(self.strikes_call) * len(self.maturities), device=self.device)
        price_vanilla_cv = torch.zeros(len(self.maturities), len(self.strikes_call), device=self.device)
        var_price_vanilla_cv = torch.zeros_like(price_vanilla_cv)

        cv_exotics = torch.zeros(S_old.shape[0], 1, device=self.device)
        exotic_option_price = torch.zeros_like(S_old)

        for i in range(1, ind_T + 1):
            idx = (i - 1) // period_length
            t = torch.ones_like(S_old) * self.timegrid[i - 1]
            h = self.timegrid[i] - self.timegrid[i - 1]
            dW = (torch.sqrt(h) * z[:, i - 1]).reshape(MC_samples, 1)

            # Compute local volatility from NN
            diffusion = self.diffusion.forward_idx(idx, torch.cat([t, S_old], 1))  # σ(S,t)
            V_new = diffusion ** 2  # Define variance as σ²(S,t)

            # Update stock price using Euler scheme
            S_new = S_old + self.rate*S_old*h/(1+self.rate*S_old.detach()*torch.sqrt(h)) + S_old*diffusion* dW/(1+S_old.detach()*diffusion.detach()*torch.sqrt(h))
            cv_vanilla += torch.exp(-self.rate * self.timegrid[i-1]) * S_old.detach() * diffusion.detach() * self.control_variate_vanilla.forward_idx(idx,torch.cat([t,S_old.detach()],1)) * dW.repeat(1,len(self.strikes_call)*len(self.maturities))
            cv_exotics += torch.exp(-self.rate * self.timegrid[i-1]) * S_old.detach() * diffusion.detach() * self.control_variate_exotics.forward_idx(idx,torch.cat([t,path],1)) * dW 
            S_old = S_new
            path[:,i] = S_old.detach().squeeze(1)
            var_path[:, i] = V_new.detach().squeeze(1)  # Store computed variance

            running_max = torch.max(running_max, S_old)

            if i in self.maturities:
                ind_maturity = self.maturities.index(i)
                for idx, strike in enumerate(self.strikes_call):
                    cv = cv_vanilla.view(-1,len(self.maturities), len(self.strikes_call))
                    price_vanilla = torch.exp(-self.rate*self.timegrid[i])*torch.clamp(S_old-strike,0).squeeze(1)-cv[:,ind_maturity,idx]
                    price_vanilla_cv[ind_maturity,idx] = price_vanilla.mean()#torch.exp(-rate/n_steps)*price.mean()
                    var_price_vanilla_cv[ind_maturity,idx] = price_vanilla.var()

        exotic_option_price = running_max - S_old
        error = torch.exp(-self.rate*self.timegrid[ind_T])*exotic_option_price.detach() - torch.mean(torch.exp(-self.rate*self.timegrid[ind_T])*exotic_option_price.detach()) - cv_exotics.detach()
        exotic_option_price = torch.exp(-self.rate*self.timegrid[ind_T])*exotic_option_price  - cv_exotics

        # Return both stock path and variance path
        return path, var_path, diffusion.detach(), price_vanilla_cv, var_price_vanilla_cv, exotic_option_price, exotic_option_price.mean(), exotic_option_price.var(), error



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.5)


def train_nsde(model, z_test, config):
    loss_fn = nn.MSELoss()
    maturities = config['maturities']
    device = config['device']
    S0 = config['init_stock']
    save_dir = config['save_dir']

    model = model.to(device)
    model.apply(init_weights)
    params_SDE = list(model.diffusion.parameters())
    n_epochs = config["n_epochs"]
    T = config["maturities"][-1]

    # we take the target data that we are interested in.
    target_mat_T = torch.tensor(config["target_data"][:len(config["maturities"]), :len(config["strikes_call"])],
                                device=device).float()

    optimizer_SDE = torch.optim.Adam(params_SDE, lr=0.001)
    optimizer_CV = torch.optim.Adam(
        list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()), lr=0.001)
    scheduler_SDE = torch.optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[500, 800], gamma=0.2)

    loss_val_best = 10
    itercount = 0

    for epoch in range(n_epochs):

        # We alternate Neural SDE optimisation and Hedging strategy optimisation
        requires_grad_CV = (epoch + 1) % 2 == 0
        requires_grad_SDE = not requires_grad_CV
        if requires_grad_CV:
            model.control_variate_vanilla.unfreeze()
            model.control_variate_exotics.unfreeze()
            model.diffusion.freeze()
        else:
            model.diffusion.unfreeze()
            model.control_variate_vanilla.freeze()
            model.control_variate_exotics.freeze()

        print('epoch:', epoch)

        batch_size = config["batch_size"]

        # we go through an epoch, i.e. 20*batch size paths
        for i in range(0, 20 * batch_size, batch_size):
            batch_z = torch.randn(batch_size, config["n_steps"],
                                  device=device)  # just me being paranoid to be sure that we have independent samples in the batch. Sampling from an antithetic dataset does not make sense to me

            optimizer_SDE.zero_grad()
            optimizer_CV.zero_grad()

            init_time = time.time()
            path, var_path, diffusion, pred, var, _, exotic_option_price, exotic_option_var, _ = model(S0, batch_z, batch_size, T,
                                                                                       period_length=16)
            time_forward = time.time() - init_time

            itercount += 1
            if requires_grad_CV:  # this chunk trains the hedging strategy
                loss = var.sum() + exotic_option_var
                init_time = time.time()
                loss.backward()
                time_backward = time.time() - init_time
                print('iteration {}, sum_variance={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount,
                                                                                                            loss.item(),
                                                                                                            time_forward,
                                                                                                            time_backward))
                nn.utils.clip_grad_norm_(
                    list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()),
                    3)
                optimizer_CV.step()
            else:
                MSE = loss_fn(pred, target_mat_T)
                loss = MSE
                init_time = time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(params_SDE, 5)
                time_backward = time.time() - init_time
                print(
                    'iteration {}, loss={:4.2e}, exotic price={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(
                        itercount, loss.item(), exotic_option_price, time_forward, time_backward))
                optimizer_SDE.step()

        scheduler_SDE.step()

        # evaluate and print RMSE validation error at the start of each epoch
        with torch.no_grad():
            path, var_path, diffusion, pred, _, exotic_option_price, exotic_price_mean, exotic_price_var, error = model(S0, z_test,
                                                                                             z_test.shape[0], T,
                                                                                             period_length=16)
            print("pred:", pred)
            print("target", target_mat_T)

        # Exotic option price hedging strategy error
        # create save directory if not exist
        save_dir_norm = os.path.normpath(save_dir)
        log_save_dir = os.path.join(save_dir_norm, "log_LV")
        if log_save_dir and not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir, exist_ok=True)
        error_hedge = error
        error_hedge_2 = torch.mean(error_hedge ** 2)
        error_hedge_inf = torch.max(torch.abs(error_hedge))
        error_path = os.path.join(save_dir, "error_hedge_LV.txt")
        with open(error_path, "a") as f:
            f.write("{},{:.4f},{:.4f},{:.4f}\n".format(epoch, error_hedge_2, error_hedge_inf, exotic_price_var.item()))
        if (epoch + 1) % 100 == 0:
            tar_path = os.path.join(save_dir, "error_hedge_LV.pth.tar")
            torch.save(error_hedge, tar_path)

        # Evaluation Error of calibration to vanilla option prices
        MSE = loss_fn(pred, target_mat_T)  # Erica: Need to change this line here to our own loss function
        loss_val = torch.sqrt(MSE)
        print('epoch={}, loss={:.4f}'.format(epoch, loss_val.item()))
        log_path = os.path.join(log_save_dir, "log_train.txt")
        with open(log_path, "a") as f:
            f.write('epoch={}, loss={:.4f}\n'.format(epoch, loss_val.item()))

        # save checkpooint
        if loss_val < loss_val_best:
            model_best = model
            loss_val_best = loss_val
            print('loss_val_best', loss_val_best)
            type_bound = "no"  # "lower" if args.lower_bound else "upper"
            # this seems to have saved the model
            filename = "Neural_SDE_exp{}_{}bound_maturity{}_AugmentedLagrangian.pth.tar".format(config["experiment"],
                                                                                                type_bound, T)
            filename = os.path.join(log_save_dir, filename)
            checkpoint = {"state_dict": model.state_dict(),
                          "exotic_price_mean": exotic_price_mean,
                          "exotic_price_var": exotic_price_var,
                          "T": T,
                          "pred": pred,
                          "target_mat_T": target_mat_T}

            torch.save(checkpoint, filename)

        if loss_val.item() < 2e-5:
            break
    return model_best