from tqdm import tqdm 
from .misc import AverageMeter
from .eval import accuracy 
from .eval import siemese_acc
import numpy as np 
from timeit import default_timer as timer
import torch

def train(model, trainloader, optimizer, criterion, device):
    t0 = timer()
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.train()
    
    pbar = tqdm(desc="training loop", total=len(trainloader.dataset), dynamic_ncols=True)
    for image, label in trainloader:
        image = image.to(device)
        label = label.to(device, non_blocking=True)

        # get output of model
        output = model(image)
        loss = criterion(output, label)

        # record accuracy and loss of the batch 
        acc = accuracy(output, label)
        losses.update(loss.item(), label.size(0))
        accuracies.update(acc, label.size(0))

        # compute gradient, do backprop and step optimizer
        optimizer.zero_grad()
        loss.backward()  # compute gradient of the loss with respect to model parameters
        optimizer.step() # calling the step function on an Optimizer makes an update to its parameters

        pbar.update(label.size(0))
    
    pbar.close()
    t1 = timer()

    return (losses.avg, accuracies.avg, t1-t0)

def siemese_train(model, trainloader, optimizer, criterion, device):
    t0 = timer()
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.train()
    
    pbar = tqdm(desc="training loop", total=len(trainloader.dataset), dynamic_ncols=True)
    for image1, image2, label in trainloader:
        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device, non_blocking=True)

        # get output of model
        output = model(image1, image2)
        loss = criterion(output.squeeze(), label.float())

        # record accuracy and loss of the batch 
        acc = siemese_acc(output.squeeze(), label)
        losses.update(loss.item(), label.size(0))
        accuracies.update(acc, label.size(0))

        # compute gradient, do backprop and step optimizer
        optimizer.zero_grad()
        loss.backward()  # compute gradient of the loss with respect to model parameters
        optimizer.step() # calling the step function on an Optimizer makes an update to its parameters

        pbar.update(label.size(0))
    
    pbar.close()
    t1 = timer()

    return (losses.avg, accuracies.avg, t1-t0)