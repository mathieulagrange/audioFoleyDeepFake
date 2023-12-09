import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F



def is_early_stopping(scheduler,loss,optimizer,stopping_rate,elder_learning_rate,condition='lr'):
    scheduler.step(loss)
    learning_rate = optimizer.param_groups[0]['lr']
    if condition=='lr': 
        stop = (learning_rate < stopping_rate)
    if condition=='improvement':
        stop = (learning_rate != elder_learning_rate)
    else :
        return False
    return stop


def run_model(model, optimizer, loader, loss_function, device, timings=None, mode='train'):
    model.train() if mode == 'train' else model.eval()

    losses = []
    accuracies = []

    
    timer_starter = torch.cuda.Event(enable_timing=True)
    timer_ender = torch.cuda.Event(enable_timing=True)

    for data, target in loader:
        data_batch_shape = data.shape[0]
        data = torch.mean(data, dim=1).view(data_batch_shape, 1, 128)
        data, target = data.to(device), target.to(device)

        if timings is not None:
            timer_starter.record(torch.cuda.Stream())

        with torch.set_grad_enabled(mode == 'train'):
            output = model(data)
            loss = loss_function(output, target)

        if timings is not None:
            timer_ender.record(torch.cuda.Stream())
            torch.cuda.synchronize()
            curr_time = timer_starter.elapsed_time(timer_ender)
            timings.append(curr_time)

        preds = (torch.sigmoid(output) > 0.5) * 1
        accuracy = torch.mean((preds == target).type(torch.float))

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses, accuracies

from sklearn.metrics import accuracy_score, confusion_matrix

def run_model_eval(model, loader, device, timings=None):
    model.eval()
    model.to(device)

    all_preds = []
    all_true_labels = []

    timer_starter = torch.cuda.Event(enable_timing=True)
    timer_ender = torch.cuda.Event(enable_timing=True)
    
    for data, target in loader:
        data_batch_shape = data.shape[0]
        data = torch.mean(data, dim=1).view(data_batch_shape, 1, 128)
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            timer_starter.record(torch.cuda.Stream())
            output = model(data)
            timer_ender.record(torch.cuda.Stream())
            
            torch.cuda.synchronize(device)
            
            curr_time = timer_starter.elapsed_time(timer_ender)
            if timings is not None:
                timings.append(curr_time)
            
            preds = (torch.sigmoid(output) > 0.5)*1
            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true_labels = np.array(all_true_labels)

    if all_true_labels.ndim > 1 and all_true_labels.shape[1] > 1:
        all_true_labels = np.argmax(all_true_labels, axis=1)

    accuracy = accuracy_score(all_true_labels, all_preds)
    conf_matrix = confusion_matrix(all_true_labels, all_preds)

    return accuracy, conf_matrix








