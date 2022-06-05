import numpy as np
import torch
#import copy

def train_1_epoch(model, device, criterion, optimizer, dataloader):
    """
    train a model for 1 epoch

    input:
        model     : instance of the model to train
        device    : 'cuda' or 'cpu'
        criterion : loss function
        optimizer : optimizer
        dataloader: PyTorch DataLoader
    
    output: 
        per-epoch accuracy
        per-epoch loss
    """
    # set model to training mode
    model.train()

    epoch_loss = 0.
    correct_preds, total_preds = 0., 0.

    for i, (images, labels) in enumerate(dataloader):
        # pass input tensors to GPU
        images, labels = images.to(device), labels.to(device)
        # zero optimizer gradients
        optimizer.zero_grad()
        # feed data to model
        outputs = model(images)
        #print(outputs) # tmp
        #print(labels)  # tmp
        # compute loss
        loss = criterion(outputs, labels)
        #print(loss) # tmp
        # update gradient
        loss.backward()
        optimizer.step()
        # get predictions
        #preds = []
        #for i in outputs:
        #    tmp = [0.] * 3
        #    max_tmp_idx = np.argmax(i.cpu().detach().numpy())
        #    tmp[max_tmp_idx] = 1.
        #    preds.append(tmp)
        #preds = torch.tensor(preds).long()
        #preds = preds.to(device)
        preds = (outputs>0.5).long()
        #print(outputs)
        #print(preds)
        # store number of correct predictions
        correct_preds += (preds==labels).sum().item()
        total_preds += torch.tensor(preds.shape).prod().item()

        epoch_loss += loss

    return correct_preds / total_preds, epoch_loss / len(dataloader)
    
def evaluate_1_epoch(model, device, criterion, dataloader):
    """
    evaluate a model for 1 epoch

    input:
        model     : instance of the model to evaluate
        device    : 'cuda' or 'cpu'
        criterion : loss function
        dataloader: PyTorch DataLoader

    output:
        per-epoch accuracy
        per-epoch loss
    """
    # set model to evaluation mode
    model.eval()

    epoch_loss = 0.
    correct_preds, total_preds = 0., 0.

    for i, (images, labels) in enumerate(dataloader):
        # pass inpute tensors to GPU
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad(): # disable autograd engine and stop weight update
            # feed data to model
            outputs = model(images)
        # compute loss
        loss = criterion(outputs, labels)
        epoch_loss += loss
        # get predictions
        #preds = []
        #for i in outputs:
        #    tmp = [0.] * 3
        #    max_tmp_idx = np.argmax(i.cpu().detach().numpy())
        #    tmp[max_tmp_idx] = 1.
        #    preds.append(tmp)
        #preds = torch.tensor(preds).long()
        #preds = preds.to(device)
        preds = (outputs>0.5).long()
        # store number of correct predictions
        correct_preds += (preds==labels).sum().item()
        total_preds += torch.tensor(preds.shape).prod().item()

    return correct_preds / total_preds, epoch_loss / len(dataloader)

def predict(model, device, dataloader):
    predictions = []
    for i, (images, labels) in enumerate(dataloader):
        # pass image tensors to device
        images = images.to(device)
        # feed data to model
        with torch.no_grad():
            outputs = model(images)
        #_, pred_label = torch.max(output, 1)
        preds = (outputs>0.5).long()
        preds = preds.cpu().numpy()
        
        predictions.extend(preds)
    
    return predictions