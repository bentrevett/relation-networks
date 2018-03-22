import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class ConvInputModel(nn.Module):
    """
    4 x Conv -> ReLU -> BN
    """
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

  
class FCOutputModel(nn.Module):
    """
    2 x Linear going from 256 to 10
    Has dropout and log_softmax
    """
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1) #is dim=1 correct? yes, because F.softmax(x, dim=1) = bs.
        return x

  

class BasicModel(nn.Module):
    """
    Basic model is just a wrapper for the train/test loop
    """
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad() #zero gradient
        output = self(input_img, input_qst) #pass img and questions through model
        loss = F.nll_loss(output, label) #calculate loss
        loss.backward() #calculate gradients
        self.optimizer.step() #update paramters
        pred = output.data.max(1)[1] #argmax prediction
        correct = pred.eq(label.data).cpu().sum() #calculate accuracy
        accuracy = correct * 100. / len(label) #accuracy into %
        return accuracy
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst) #pass img and questions through model
        pred = output.data.max(1)[1] #argmax prediction
        correct = pred.eq(label.data).cpu().sum() #calculate accuracy
        accuracy = correct * 100. / len(label) #accuracy into %
        return accuracy

    def save_model(self, epoch):
        #save model parameters (state_dict)
        torch.save(self.state_dict(), 'checkpoints/{}_{:02d}.pt'.format(self.name, epoch))


class RN(BasicModel):
    """
    Instance of basic model with name == 'RN'
    """
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.conv = ConvInputModel() #4 x Conv -> ReLU -> BN
        
        #(number of filters per object+coordinate of object)*2+question vector
        # output of conv is 24x5x5
        # +2 is for the x, y co-ordinates of the object
        # *2 because you concat each object to every other object
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        #co-ordinates for objects i and j
        #bs, dims (x and y)
        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        # x goes from -1 to 1.4, y goes from -1 to 1 in steps of 0.5 and then back to -1
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        self.fcout = FCOutputModel() # 2 x Linear from 256 to 10
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        # 25 is from the 5x5 size image, each pixel is now a co-ordinate
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        # x_flat is (64 x 25 x 26)
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)
        
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1,25,1)
        qst = torch.unsqueeze(qst, 2)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
        x_i = x_i.repeat(1,25,1,1) # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)
        x_j = torch.cat([x_j,qst],3)
        x_j = x_j.repeat(1,1,25,1) # (64x25x25x26+11)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)
        
        # reshape for passing through network
        x_ = x_full.view(mb*d*d*d*d,63)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb,d*d*d*d,256)
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)


class CNN_MLP(BasicModel):
    """
    Instance of basic model with name == 'CNNMLP'
    """
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel() #goes from 3x75x75 to 24x5x5
        self.fc1   = nn.Linear(24*5*5 + 11, 256) #img o/p after conv is 24x5x5, 11 is from questions
        self.fcout = FCOutputModel() #goes from 256 to 10

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1) #reshape from (5x5x24 to 600)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)