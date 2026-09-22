# external imports
import copy
import math
import numpy as np
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter 
from torch.utils.data import TensorDataset, DataLoader

def print_torch_architecture(network):
    print("Network modules:")
    for i in network.named_modules():
        if i[0] != '':
            print(i)
    print(" ")

def print_count_torch_parameters(network):
    n_parameters_total = 0
    print("Number of trainable parameters:")
    for name, param in network.named_parameters():
        print(name + ":", param.numel(), "parameters")
        n_parameters_total += param.numel()
    print("Total number of trainable parameters: ", str(n_parameters_total))
    print(" ")
    return

def print_torch_parameters(network, extend = False):
    print("Network parameters:")
    for name, param in network.named_parameters():
        if extend:
            print(name, "is a", param)
            print(" ")
        else:
            print(name + ":", param.data)
    if not extend:
        print(" ")
    return

def train_no(
        my_no,
        disp_nn_data,
        force_nn_data,
        Ibar_nn_data,
        Wbar_nn_data,
        hyperparameters_training,
        _verbose=True,
        trial=None
        ):
    
    if trial is not None: import optuna

    # unpack hyperparameters
    n_epochs = hyperparameters_training["n_epochs"]
    batch_size = hyperparameters_training["batch_size"]
    lr = hyperparameters_training["lr"]
    weight_decay = hyperparameters_training["weight_decay"]
    optimizer_name = hyperparameters_training["optimizer_name"]
    scheduler_name = hyperparameters_training["scheduler_name"]
    T0 = hyperparameters_training["T0"]
    T_mult = hyperparameters_training["T_mult"]
    eta_min = hyperparameters_training["eta_min"]
    idx_train = hyperparameters_training["idx_train"]
    idx_val = hyperparameters_training["idx_val"]

    # split training data
    displacement_train = disp_nn_data[idx_train]
    force_train = force_nn_data[idx_train]
    Ibar_train = Ibar_nn_data[idx_train]
    Wbar_train = Wbar_nn_data[idx_train]

    displacement_val = disp_nn_data[idx_val]
    force_val = force_nn_data[idx_val]
    Ibar_val = Ibar_nn_data[idx_val]
    Wbar_val = Wbar_nn_data[idx_val]

    # loss function
    loss_fn = nn.MSELoss()

    # optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(my_no.parameters(), lr=lr, weight_decay=weight_decay)
    
    # scheduler
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=T_mult, eta_min=eta_min)
    
    if batch_size is None:

        # training
        if _verbose: print("\nTraining:")
        train_loss_vec = np.zeros(n_epochs)
        val_loss_vec = np.zeros(n_epochs)
        best_loss = float("inf")
        best_state_dict = None
        total_start = time.time()
        epoch_start = time.time()
        for epoch in range(n_epochs):

            # training step
            my_no.train()
            optimizer.zero_grad()
            Wbar_pred_train = my_no(displacement_train, force_train, Ibar_train)
            loss_train = loss_fn(Wbar_pred_train, Wbar_train)
            loss_train.backward()
            optimizer.step()
            if scheduler_name == "CosineAnnealingWarmRestarts":
                scheduler.step()
            train_loss_vec[epoch] = loss_train.item()

            # validation step
            my_no.eval()
            with torch.no_grad():
                Wbar_pred_val = my_no(displacement_val, force_val, Ibar_val)
                loss_val = loss_fn(Wbar_pred_val, Wbar_val)
                val_loss_vec[epoch] = loss_val.item()
                if loss_val.item() < best_loss:
                    best_loss = loss_val.item()
                    best_state_dict = copy.deepcopy(my_no.state_dict())

            # verbose
            if _verbose and epoch-1 % 100 == 0:
                epoch_end = time.time()
                print(f"Epoch {epoch}/{n_epochs} | Train loss: {loss_train.item():.6f} | Validation loss: {loss_val.item():.6f} | Time: {epoch_end - epoch_start:.2f}s")
                epoch_start = time.time()

        total_end = time.time()
        if _verbose: print(f"Total training time: {total_end - total_start:.2f}s")

        return best_state_dict, train_loss_vec, val_loss_vec
    
    else:

        train_dataset = TensorDataset(
            displacement_train,
            force_train,
            Ibar_train,
            Wbar_train
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        # training
        print("\nTraining:")
        train_loss_vec = np.zeros(n_epochs)
        val_loss_vec = np.zeros(n_epochs)
        best_loss = float("inf")
        best_state_dict = None
        total_start = time.time()
        epoch_start = time.time()
        for epoch in range(n_epochs):
            
            # training step
            my_no.train()
            epoch_train_loss = 0.0
            for batch_idx, (disp_batch, force_batch, Ibar_batch, Wbar_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                Wbar_pred_train = my_no(disp_batch, force_batch, Ibar_batch)
                loss_batch = loss_fn(Wbar_pred_train, Wbar_batch)
                loss_batch.backward()
                optimizer.step()
                if scheduler_name == "CosineAnnealingWarmRestarts":
                    scheduler.step(epoch + batch_idx / len(train_loader))
                epoch_train_loss += loss_batch.item() * disp_batch.size(0)
            epoch_train_loss /= len(train_dataset)
            train_loss_vec[epoch] = epoch_train_loss

            # validation step
            my_no.eval()
            with torch.no_grad():
                Wbar_pred_val = my_no(displacement_val, force_val, Ibar_val)
                loss_val = loss_fn(Wbar_pred_val, Wbar_val)
                val_loss_vec[epoch] = loss_val.item()
                if loss_val.item() < best_loss:
                    best_loss = loss_val.item()
                    best_state_dict = copy.deepcopy(my_no.state_dict())

            if trial is not None:
                trial.report(loss_val.item(), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            # verbose
            if epoch % 100 == 0:
                epoch_end = time.time()
                print(f"Epoch {epoch}/{n_epochs} | Train loss: {epoch_train_loss:.6f} | Validation loss: {loss_val.item():.6f} | Time: {epoch_end - epoch_start:.2f}s")
                epoch_start = time.time()
            
        total_end = time.time()
        print(f"Total training time: {total_end - total_start:.2f}s")

        return best_state_dict, train_loss_vec, val_loss_vec



