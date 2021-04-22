# !pip3 install fairseq
# !pip3 install fastbpe
# !pip3 install vncorenlp
# !pip3 install transformers

import os
import time
import pickle
from tqdm.auto import tqdm
from os.path import join
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from transformers.modeling_utils import * 
from transformers import AdamW

import config
from utils import *
from models import ClassifierModel

# Load data
print("### Loading data")
X = load_pkl(config.PATH['X_ENCODE'])
y = load_pkl(config.PATH['Y_ENCODE'])

# Get device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model
print("### Loading model")
classifier = ClassifierModel(device = DEVICE)
model = classifier.bert_model

# Freeze model
for child in model.children():
    for param in child.parameters():
        param.requires_grad = False

for param in model._modules['model']._modules['classification_heads']._modules['new_task'].parameters():
    param.requires_grad = True           

EPOCHS = config.TRAIN['EPOCHS']
BATCH_SIZE = config.TRAIN['BATCH_SIZE']
FOLD = config.TRAIN['FOLD']
LR = config.TRAIN['LEARNING_RATE']


# Define optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=LR, correct_bias=False)  # To reproduce 
criteria = nn.CrossEntropyLoss()

# Prepare dataset
splits = list(StratifiedKFold(n_splits=5, shuffle=True).split(X, y))
train_idx, val_idx = splits[FOLD]
# Create dataset
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X[train_idx],dtype=torch.long), torch.tensor(y[train_idx],dtype=torch.long))
valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X[val_idx],dtype=torch.long), torch.tensor(y[val_idx],dtype=torch.long))

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate(logits, targets):
    """
    Đánh giá model sử dụng accuracy và f1 scores.
    Args:
        logits (B,C): torch.LongTensor. giá trị predicted logit cho class output.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        acc (float): the accuracy score
        f1 (float): the f1 score
    """
    logits = logits.detach().cpu().numpy()    
    y_pred = np.argmax(logits, axis = 1)
    targets = targets.detach().cpu().numpy()
    f1 = f1_score(targets, y_pred, average='weighted')
    acc = accuracy_score(targets, y_pred)
    return acc, f1

def validate(valid_loader, model, device):
    model.eval()
    accs = []
    f1s = []
    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model.predict('new_task', x_batch)
            logits = torch.exp(outputs)
            acc, f1 = evaluate(logits, y_batch)
            accs.append(acc)
            f1s.append(f1)
    
    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    return mean_acc, mean_f1

# load pretrained model
last_epoch = 0
if os.path.exists(config.PATH['PRETRAIN']):
    checkpoint = torch.load(config.PATH['PRETRAIN'], map_location = DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = int(checkpoint['epoch'])

display_step = 100
total_loss = []
for epoch in range(last_epoch, EPOCHS):
    # Train model
    print("Epoch ", epoch)
    model.train()
    start = time.time()
    for step, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
        x_batch = x_batch.to(device = DEVICE)
        y_batch = y_batch.to(device = DEVICE)
        
        optimizer.zero_grad()
        y_logits = model.predict('new_task', x_batch)
        loss = criteria(y_logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())
        if step % display_step == 0 and step > 0:
            print(f'Epoch {epoch}: step {step}: loss {sum(total_loss[-display_step:]) / display_step}')
    
    # save model
    ckpt_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt_dict, config.PATH['PRETRAIN'])

    # validate
    acc, f1 = validate(valid_loader, model, DEVICE)
    print(f'Epoch {epoch} Accuray {acc:.4f} F1 score: {f1:.4f} in {time.time() - start} ms')
    start = time.time()