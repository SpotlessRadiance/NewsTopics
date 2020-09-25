import torch
import copy
from torch import nn
from torch.utils.data import DataLoader


def train_model(model, epochs, train_dataset, val_dataset, batch_size, lr, criterion, optimizer=None,
                epoch_patience=5, lr_scheduler = None, model_path = None, device = torch.cuda('cpu')):
    '''Train model and save model with best val score in model_path
    :returns model with best score on validation dataset
             best score on validation dataset'''
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optimizer(model.parameters(), lr=lr)
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)
    for epoch in range(epochs):
        train_loss = 0
        batches = 0
        for i, (x_batch, y_batch) in enumerate(train_data):
            x = x_batch.to(device)
            y = y_batch.to(device)
            y = torch.squeeze(y)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= (i+1)
        print(f'Epoch {epoch}, training loss = {train_loss} ')
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(val_data):
                x = x_batch.to(device)
                y = y_batch.to(device)
                y = torch.squeeze(y)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
            val_loss /= (i+1)
            print(f'val loss = {val_loss}')
            if best_val_loss > val_loss:
                if model_path != None:
                    torch.save(model.state_dict(), model_path)
                best_model = copy.copy(model)
                best_epoch = epoch
                best_val_loss = val_loss
            elif epoch - best_epoch > epoch_patience:
                print(f'Early stopping')
                return best_val_loss, best_model
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)
    return best_val_loss, best_model