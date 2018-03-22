"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""

import argparse
import os
import pickle
import random
import numpy as np

import torch
from torch.autograd import Variable

from models import RN, CNN_MLP

def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, rel, norel):
    model.train() #puts model in training mode

    #rel and norel are a list of tuples with: (img, q, a)
    #currently they are ordered together, so need to shuffle
    random.shuffle(rel)
    random.shuffle(norel)

    #data goes from a list of tuples of (img, q, a)
    #to a tuple of lists ([imgs, qs, as])
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    for batch_idx in range(len(rel[0]) // bs):
        #slice into data and convert to tensors
        tensor_data(rel, batch_idx)
        #input_img = [bs, c, h, w]
        #input_qst = [bs, n_q]
        #label = [bs]
        accuracy_rel = model.train_(input_img, input_qst, label)

        tensor_data(norel, batch_idx)
        accuracy_norel = model.train_(input_img, input_qst, label)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(epoch, batch_idx * bs * 2, len(rel[0]) * 2, \
                                                                                                                           100. * batch_idx * bs/ len(rel[0]), accuracy_rel, accuracy_norel))
            

def test(epoch, rel, norel):
    model.eval()
    
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        accuracy_rels.append(model.test_(input_img, input_qst, label))

        tensor_data(norel, batch_idx)
        accuracy_norels.append(model.test_(input_img, input_qst, label))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
        accuracy_rel, accuracy_norel))

def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    #data is a tuple 
    #first element is a list of tuples
    #first element of that tuple is the image
    #second element of that tuple is the "relation" questions
    #third element of that tuple is the "norelation" questions
    #questions are tuples where first element is the question and second element is the answer
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    rel_train = [] #list of tuples: (img, q, a)
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img,0,2) #data is loaded at HxWxC, want CxHxW
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (rel_train, rel_test, norel_train, norel_test)

def tensor_data(data, i):
    #data is tuple of lists with (imgs, qs, as)
    #convert from numpy arrays to torch tensors
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    #don't need to reshape the input/label tensors, but do it anyway
    #set the data of the input/label tensors to be copy of the img/qst/ans tensors
    #not sure why
    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#CNN_MLP is baseline model
#RN is relation network model
if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
else:
  model = RN(args)
  
#create tensors to hold data to feed into model
bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75) #image is 3 channels and size 75x75
input_qst = torch.FloatTensor(bs, 11) #question is 11-dim vector
label = torch.LongTensor(bs)

#place on GPU
if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

#convert to Variables
input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

#load data
#each is a list of tuples with: (img, q, a)
#img = CxHxW = (3x75x75) np array
#q = (11,) np array
#a = int
rel_train, rel_test, norel_train, norel_test = load_data()

#for saving model parameters
model_dirs = './checkpoints'
try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

#if resuming, load from exact model parameter name
if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

#do train/test loop and save at the end of every epoch
for epoch in range(1, args.epochs + 1):
    train(epoch, rel_train, norel_train) #train epoch
    test(epoch, rel_test, norel_test) #test epoch
    model.save_model(epoch) #saves to checkpoints/<model_name>_<epoch_n>.pt