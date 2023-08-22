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


def find_lr(
    model,
    train_loader,
    optimizer="adam",
    loss_fn="crossentropy",
    device_type="CPU",
    init_value=1e-8,
    final_value=1e1,
):

    #     https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

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

    loss_fns = {
        "crossentropy": torch.nn.CrossEntropyLoss,
        "multimargin": torch.nn.MultiMarginLoss,
        "softmargin": torch.nn.SoftMarginLoss,
        "nnl": torch.nn.NLLLoss,
    }

    optimizers = {
        "adamw": optim.AdamW,
        "adam": optim.Adam,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "rmsprop": optim.RMSprop,
        "sgd": optim.SGD,
        "rprop": optim.Rprop,
    }

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

        if batch_num > 1 and loss > 4e1 * best_loss:
            losses.append(loss)
            log_lrs.append(lr)
            return (log_lrs[10:], losses[10:]) if len(log_lrs) > 20 else (log_lrs, losses)
        # record the best loss

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        losses.append(loss)
        #         log_lrs.append(math.log10(lr))

        log_lrs.append(lr)

        loss.backward()
        optimizer.step()

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr

    losses = moving_average(losses, 5)

    return (log_lrs[10:], losses[10:]) if len(log_lrs) > 20 else (log_lrs, losses)


def train(
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
    Train pytorch model
    
    :param model: Pytorch model
    :returns: Metrics dataframe
    """

    if (device_type == "GPU") & (torch.cuda.is_available()):
        device = torch.device("cuda")
        torch.cuda.empty_cache()  # clear data
        print(f"Training on GPU...\n")
    elif device_type == "CPU":
        device = torch.device("cpu")
        print(f"Training on CPU...\n")
    elif (device_type == "GPU") & (not torch.cuda.is_available()):
        raise Exception("""GPU not found""")
    else:
        raise Exception("""Please choose between 'CPU' and 'GPU' for device type""")

    model = model.to(device)

    loss_fns = {
        "crossentropy": torch.nn.CrossEntropyLoss,
        "multimargin": torch.nn.MultiMarginLoss,
        "softmargin": torch.nn.SoftMarginLoss,
        "nnl": torch.nn.NLLLoss,
    }

    optimizers = {
        "adamw": optim.AdamW,
        "adam": optim.Adam,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "rmsprop": optim.RMSprop,
        "sgd": optim.SGD,
        "rprop": optim.Rprop,
    }

    # Instantiate loss function and optimizer
    # TODO: error catching for not implemented

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

        for batch in tqdm(train_loader, desc="Training Batch", leave=epoch == epochs):  #

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

        for batch in tqdm(val_loader, desc="Validation Batch", leave=epoch == epochs):

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
        metrics_dict["Training Accuracy"].append(training_accuracy)
        metrics_dict["Validation Accuracy"].append(validation_accuracy)
            #### PRINT PERFORMANCE METRICS #####

    metrics_dict = pd.DataFrame(metrics_dict)

    return model, metrics_dict

