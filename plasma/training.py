from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.models as models
from tqdm.notebook import tqdm

import math


def plasma_train(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer="adam",
    loss_fn="crossentropy",
    epochs: int = 20,
    learning_rate=3e-4,
    device_type: str = "cpu",
) -> pd.DataFrame:
    """
    Train pytorch model, return metrics dataframe


    :param model: Pytorch model
    :returns: Metrics dataframe
    """

    if (device_type == "GPU") & (torch.cuda.is_available()):
        device = torch.device("cuda")
        print(f"Training on GPU...\n")
    elif device_type == "CPU":
        device = torch.device("cpu")
        print(f"Training on CPU...\n")
    elif (device_type == "GPU") & (not torch.cuda.is_available()):
        raise Exception("""GPU not found""")
    else:
        raise Exception("""Please choose between 'CPU' and 'GPU' for device type""")

    model = model.to(device)

    loss_fns = {"crossentropy": torch.nn.CrossEntropyLoss}

    optimizers = {"adam": optim.Adam}

    # Instantiate loss function and optimizer
    # !TODO: error catching for not implemented

    loss_fn, optimizer = (
        loss_fns[loss_fn](),
        optimizers[optimizer](model.parameters(), lr=learning_rate),
    )

    metrics_dict = {
        "Epoch": [],
        "Training Loss": [],
        "Validation Loss": [],
        "Training Accuracy": [],
        "Validation Accuracy": [],
    }

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):

        #### TRAINING LOOP ####
        training_loss = 0.0

        model.train()

        num_correct = 0
        num_training_examples = 0

        for batch in tqdm(
            train_loader, desc="Training Batch", leave=bool(epoch == epochs)
        ):  #

            optimizer.zero_grad()  # zeroize gradients

            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)  # forward pass

            loss = loss_fn(output, targets)  # calculate loss
            loss.backward()  # calculate gradients

            optimizer.step()  # adjust/step weights + biases

            training_loss += loss.data.item() * inputs.size()[0]

            correct = torch.eq(
                torch.max(output.softmax(dim=1), dim=-1)[1], targets.squeeze()
            ).sum()
            num_correct += correct.data.item()
            num_training_examples += inputs.shape[0]
        training_accuracy = num_correct / num_training_examples

        training_loss /= len(
            train_loader.dataset
        )  # weighted average training loss for epoch
        #### TRAINING LOOP ####

        #### VALIDATION/EVALUATION LOOP ####
        valid_loss = 0.0
        num_correct = 0
        num_validation_examples = 0

        model.eval()

        for batch in tqdm(
            val_loader, desc="Validation Batch", leave=bool(epoch == epochs)
        ):

            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)

            loss = loss_fn(output, targets)  # calculate loss
            valid_loss += loss.data.item() * inputs.size()[0]

            correct = torch.eq(
                torch.max(output.softmax(dim=1), dim=-1)[1], targets.squeeze()
            ).sum()
            num_correct += correct.data.item()
            num_validation_examples += inputs.shape[0]

        valid_loss /= len(val_loader.dataset)
        validation_accuracy = num_correct / num_validation_examples

        #### VALIDATION/EVALUATION LOOP ####

        #### PRINT PERFORMANCE METRICS #####
        metrics_dict["Epoch"].append(epoch)
        metrics_dict["Training Loss"].append(training_loss)
        metrics_dict["Validation Loss"].append(valid_loss)
        # metrics_dict["Training Accuracy"].append(training_accuracy)
        # metrics_dict["Validation Accuracy"].append(validation_accuracy)
        #### PRINT PERFORMANCE METRICS #####

    metrics_dict = pd.DataFrame(metrics_dict)

    return model, metrics_dict


def plot_history(metrics_df):
    metrics_df_ = pd.melt(
        metrics_df,
        id_vars=["Epoch"],
        value_vars=list(set(metrics_df.columns) - set(["Epoch"])),
    )

    g = sns.lineplot(x="Epoch", y="value", hue="variable", data=metrics_df_)

    plt.show()


def find_optimal_lr(
    model,
    train_loader,
    optimizer="adam",
    loss_fn="crossentropy",
    device_type="CPU",
    init_value=1e-8,
    final_value=1e1,
):

    batch_number = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / batch_number)
    lr = init_value

    # ----------------------

    if (device_type == "GPU") & (torch.cuda.is_available()):
        device = torch.device("cuda")
        print(f"Training on GPU...\n")
    elif device_type == "CPU":
        device = torch.device("cpu")
        print(f"Training on CPU...\n")
    elif (device_type == "GPU") & (not torch.cuda.is_available()):
        raise Exception("""GPU not found""")
    else:
        raise Exception("""Please choose between 'CPU' and 'GPU' for device type""")

    model = model.to(device)

    loss_fns = {"crossentropy": torch.nn.CrossEntropyLoss}

    optimizers = {"adam": optim.AdamW}

    # Instantiate loss function and optimizer
    # TODO: error catching for not implemented

    loss_fn, optimizer = (
        loss_fns[loss_fn](),
        optimizers[optimizer](model.parameters()),
    )

    # ----------------------

    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []

    for data in tqdm(train_loader, desc="Training Batch"):
        batch_num += 1
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs, losses

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        losses.append(loss)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr

    return log_lrs, losses
