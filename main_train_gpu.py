# external imports
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

rs = 42
np.random.seed(rs)
torch.manual_seed(rs)

# custom imports
import neural_operators as no

def train_gpu(
    hyperparameters_data_encoder,
    hyperparameters_no,
    hyperparameters_training,
    name_folder = None,
    _verbose = False,
    _save = True,
    trial = None
    ):

    # settings
    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S%f")
    if hyperparameters_no["name_no"] == "cano":
        hyperparameters_data_encoder["n_encode"] = 6

    # cpu / gpu settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    if torch.cuda.is_available(): print("Device name:", torch.cuda.get_device_name(0))

    # load data
    data = np.load("data/" + hyperparameters_training["name_experiment"] + "/" + hyperparameters_training["name_training"] + ".npz")
    disp_encode_nn_data = data["disp_encode_nn_data"]
    force_nn_data = data["force_nn_data"]
    Ibar_nn_data = data["Ibar_nn_data"]
    Wbar_nn_data = data["Wbar_nn_data"]
    hyperparameters_training["idx_train"] = torch.tensor(data["idx_train"], dtype=torch.int64, device=device)
    hyperparameters_training["idx_val"] = torch.tensor(data["idx_val"], dtype=torch.int64, device=device)

    # convert to torch tensors
    disp_encode_nn_data = torch.tensor(disp_encode_nn_data, dtype=torch.float32, device=device)
    force_nn_data = torch.tensor(force_nn_data, dtype=torch.float32, device=device)
    Ibar_nn_data = torch.tensor(Ibar_nn_data, dtype=torch.float32, device=device)
    Wbar_nn_data = torch.tensor(Wbar_nn_data, dtype=torch.float32, device=device)

    # print shapes
    if _verbose:
        print("\nData:")
        print("disp_encode_nn_data shape:", disp_encode_nn_data.shape)
        print("force_nn_data shape:", force_nn_data.shape)
        print("Ibar_nn_data shape:", Ibar_nn_data.shape)
        print("Wbar_nn_data shape:", Wbar_nn_data.shape)

    hyperparameters_data_encoder["n_step"] = disp_encode_nn_data.shape[1]
    hyperparameters_data_encoder["n_point"] = disp_encode_nn_data.shape[2]
    hyperparameters_data_encoder["n_dim"] = disp_encode_nn_data.shape[3]

    # neural network architecture
    if hyperparameters_no["name_no"] == "pano":
        my_no = no.pano(hyperparameters_no, hyperparameters_data_encoder).to(device)
    if hyperparameters_no["name_no"] == "cano":
        my_no = no.cano(hyperparameters_no, hyperparameters_data_encoder).to(device)
    if _verbose:
        print("\nNeural network architecture:")
        no.print_count_torch_parameters(my_no)

    # training
    best_state_dict, train_loss_vec, val_loss_vec = no.train_no(
        my_no,
        disp_encode_nn_data,
        force_nn_data,
        Ibar_nn_data,
        Wbar_nn_data,
        hyperparameters_training,
        _verbose=_verbose,
        trial=trial
        )

    if _save:
        if name_folder is None: name_folder = "models"
        else: name_folder = "models/" + name_folder
        os.makedirs(f"{name_folder}", exist_ok=True)

        # plot training and validation loss
        plt.figure()
        plt.plot(train_loss_vec, color="blue", lw=2, label="Training loss")
        plt.plot(val_loss_vec, color="red", lw=2, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()        
        plt.savefig(f"{name_folder}/{hyperparameters_no['name_no']}_{timestamp}_training.png", dpi=500, bbox_inches="tight")

        # save network
        torch.save({
            "hyperparameters_no": my_no.hyperparameters_no,
            "hyperparameters_data_encoder": my_no.hyperparameters_data_encoder,
            "hyperparameters_training": hyperparameters_training,
            "model_state": best_state_dict,
            "train_loss_vec": train_loss_vec.tolist(),
            "val_loss_vec": val_loss_vec.tolist(),
        }, f"{name_folder}/{hyperparameters_no['name_no']}_{timestamp}.pth")

    return min(val_loss_vec)


if __name__ == "__main__":
    hyperparameters_data_encoder = {
        "name_data_encoder": "data_encoder_fnn",
        "n_step": None,
        "n_point": None,
        "n_dim": None,
        "n_encode": 12,
        "n_neuron": [512],
        # "n_neuron": [2048],
        # "n_neuron": [4096],
        "activation": "ReLU",
    }
    # hyperparameters_no = {
    #     "name_no": "pano",
    #     "n_input": 1,
    #     "n_output": 1,
    #     "n_neuron": [4],
    # }
    hyperparameters_no = {
        "name_no": "cano",
    }
    hyperparameters_training = {
        "name_experiment": "plate_with_corner_hole_3D_u2",
        "name_training": "training_Taylor_Laplace_100_grid_linear_10",
        "rs": rs,
        "n_epochs": 10000,
        # "batch_size": None,
        # "batch_size": 32,
        "batch_size": 64,
        # "batch_size": 128,
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "optimizer_name": "Adam",
        "scheduler_name": "CosineAnnealingWarmRestarts",
        "T0": 1000,
        "T_mult": 1,
        "eta_min": 0,
        "idx_train": None,
        "idx_val": None,
    }

    score = train_gpu(
        hyperparameters_data_encoder,
        hyperparameters_no,
        hyperparameters_training,
        _verbose = True
    )

