import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator

from tqdm import tqdm

import sys

import models

#constants 
BATCH_SIZE = 32

#define fields
Q = Field(batch_first=True)
A = Field(batch_first=True)
SF = Field(batch_first=True)

#{json key name: (attribute name, field name)}
fields = {'q': ('question', Q), 
          'a': ('answer', A), 
          'sf0': ('sf0', SF),
          'sf1': ('sf1', SF),
          'sf2': ('sf2', SF),
          'sf3': ('sf3', SF),
          'sf4': ('sf4', SF),
          'sf5': ('sf5', SF),
          'sf6': ('sf6', SF),
          'sf7': ('sf7', SF),
          }

#get data from jsonl
train, test = TabularDataset.splits(
                path = 'data',
                train = 'train_all.jsonl',
                test = 'test_all.jsonl',
                format = 'json',
                fields = fields
)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print(vars(train[0]))

# build the vocabulary
Q.build_vocab(train.question, train.sf0, train.sf1, train.sf2, train.sf3, train.sf4, train.sf5, train.sf6, train.sf7)
A.build_vocab(train.answer)
SF.build_vocab(train.question, train.sf0, train.sf1, train.sf2, train.sf3, train.sf4, train.sf5, train.sf6, train.sf7)

# print vocab information
print('len(Q.vocab)', len(Q.vocab))
print('len(SF.vocab)', len(SF.vocab))
print('len(A.vocab)', len(A.vocab))
print('Most common tokens', Q.vocab.freqs.most_common(10))

# make iterator for splits
train_iter, test_iter = BucketIterator.splits(
    (train, test), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.sf0),
    device = None if torch.cuda.is_available() else -1,
    repeat=False)

#initialize model
model = models.RelationNetwork(len(Q.vocab),
                               len(SF.vocab),
                               len(A.vocab))

print(model)

#initialize optimizer, scheduler and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)

#place on GPU
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

best_test_loss = float('inf')
best_test_acc = 0
epoch = 0

while True:

    epoch += 1
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(train_iter, desc='Train'):

        optimizer.zero_grad()

        predictions = model(batch.question, batch.sf0, batch.sf1, batch.sf2, batch.sf3, batch.sf4, batch.sf5, batch.sf6, batch.sf7)

        loss = criterion(predictions, batch.answer.squeeze(1))

        loss.backward()

        optimizer.step()

        pred = predictions.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        acc = pred.eq(batch.answer.data.view_as(pred)).long().cpu().sum()

        epoch_loss += loss.data[0]
        epoch_acc += acc/len(pred)

    #calculate metrics averaged across whole batch
    train_loss = epoch_loss / len(train_iter)
    train_acc = epoch_acc / len(train_iter)

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()

    for batch in tqdm(test_iter, desc=' Test'):

        predictions = model(batch.question, batch.sf0, batch.sf1, batch.sf2, batch.sf3, batch.sf4, batch.sf5, batch.sf6, batch.sf7)

        loss = criterion(predictions, batch.answer.squeeze(1))
        
        pred = predictions.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        acc = pred.eq(batch.answer.data.view_as(pred)).long().cpu().sum()

        epoch_loss += loss.data[0]
        epoch_acc += acc/len(pred)

    #calculate metrics averaged across whole epoch
    test_acc = epoch_acc / len(test_iter)
    test_loss = epoch_loss / len(test_iter)

    #update scheduler
    #scheduler.step(test_loss)

    #print metrics
    print(f'Epoch: {epoch}') 
    print(f'Train Loss: {train_loss:.3f}, Train Acc.: {train_acc*100:.2f}%')
    print(f'Test Loss: {test_loss:.3f}, Test Acc.: {test_acc*100:.2f}%')

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_test_acc = test_acc
        train_patience_count = 1
    else:
        print(f'Losing patience... {train_patience_count}/3')
        train_patience_count += 1

    if train_patience_count > 3:
        print('Lost patience!')
        print(f'Best test loss: {best_test_loss}')
        print(f'Best test acc: {best_test_acc}')
        sys.exit(0)