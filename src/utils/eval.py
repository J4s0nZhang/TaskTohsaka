import torch 
from tqdm import tqdm 
from .misc import AverageMeter
import numpy as np 
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix

def accuracy(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = (pred == target).sum().item()
    return correct / output.shape[0]

def siemese_acc(output, target):
    m = torch.nn.Sigmoid()
    output = m(output)
    preds = (output > 0.5).float()
    correct = (preds == target).sum().item() 
    return correct / output.shape[0]
    
def evaluate(model, valloader, criterion, device, n_classes=3):
    t0 = timer()
    losses = AverageMeter()
    accuracies = AverageMeter()

    labels = np.zeros([len(valloader.dataset)], dtype=np.int32)
    preds = np.zeros([len(valloader.dataset)], dtype=np.int32)
    idx = 0 
    # switch to evaluate mode
    model.eval()

    pbar = tqdm(desc="val loop", total=len(valloader.dataset), dynamic_ncols=True)
    with torch.no_grad():
        for image, label in valloader: 
            image = image.to(device)
            label = label.to(device, non_blocking=True)

            output = model(image)
            loss = criterion(output, label)

            #record metrics
            acc = accuracy(output, label)
            losses.update(loss.item(), label.size(0))
            accuracies.update(acc, label.size(0)) 

            labels[idx:idx+label.size(0)] = label.cpu().numpy()
            preds[idx:idx+label.size(0)] = output.max(dim=1)[1].cpu().numpy()
            idx += label.size(0)

            pbar.update(label.size(0))
        
        pbar.close()
    
    t1 = timer() 
    confmat = confusion_matrix(labels, preds, labels=np.arange(n_classes))
    return (losses.avg, accuracies.avg, confmat, t1-t0)

def siemese_evaluate(model, valloader, criterion, device, n_classes=3):
    t0 = timer()
    losses = AverageMeter()
    accuracies = AverageMeter()

    labels = np.zeros([len(valloader.dataset)], dtype=np.int32)
    preds = np.zeros([len(valloader.dataset)], dtype=np.int32)
    idx = 0 
    # switch to evaluate mode
    model.eval()

    pbar = tqdm(desc="val loop", total=len(valloader.dataset), dynamic_ncols=True)
    with torch.no_grad():
        for image1, image2, label in valloader:
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device, non_blocking=True)

            # get output of model
            output = model(image1, image2)
            loss = criterion(torch.squeeze(output), label.float())

            #record metrics
            acc = siemese_acc(output.squeeze(), label)
            losses.update(loss.item(), label.size(0))
            accuracies.update(acc, label.size(0)) 

            labels[idx:idx+label.size(0)] = label.cpu().numpy()
            preds[idx:idx+label.size(0)] = output.max(dim=1)[1].cpu().numpy()
            idx += label.size(0)

            pbar.update(label.size(0))
        
        pbar.close()
    
    t1 = timer() 
    confmat = confusion_matrix(labels, preds, labels=np.arange(n_classes))
    return (losses.avg, accuracies.avg, confmat, t1-t0)

# WIP: deciding testing visualization pipeline 
def plot_pred(preds, batch_size, image_batch, rows=4, cols=4):
    plt.figure(figsize=(10, 10))
    for i in range(batch_size):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")

def plot_pred_batch(model, valloader, device, num=8):
    accuracies = AverageMeter()

    labels = np.zeros([num], dtype=np.int32)
    preds = np.zeros([num], dtype=np.int32)
    idx = 0 
    # switch to evaluate mode
    model.eval()

    count = 0
    with torch.no_grad():
        for image, label in valloader: 
            if count >= num:
                break
            image = image.to(device)
            label = label.to(device, non_blocking=True)

            output = model(image)

            #record metrics
            acc = accuracy(output, label)
            accuracies.update(acc, label.size(0)) 

            labels[idx:idx+label.size(0)] = label.cpu().numpy()
            preds[idx:idx+label.size(0)] = output.max(dim=1)[1].cpu().numpy()
            idx += label.size(0)
            count += label.size(0)

    