import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RelationNetwork(nn.Module):
    def __init__(self, q_vocab_size, sf_vocab_size, a_vocab_size):
        super().__init__()

        self.q_embbeding = nn.Embedding(q_vocab_size,32)
        self.sf_embedding = nn.Embedding(sf_vocab_size,32)

        self.q_rnn = nn.LSTM(32, 32)
        self.sf_rnn = nn.LSTM(32, 32)
        
        self.g1 = nn.Linear(32*3,256)
        self.g2 = nn.Linear(256,256)
        self.g3 = nn.Linear(256,256)
        self.g4 = nn.Linear(256,256)

        self.f1 = nn.Linear(256,256)
        self.f2 = nn.Linear(256,512)
        self.f3 = nn.Linear(512,a_vocab_size)

    def forward(self, q, sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7):

        q = self.q_embbeding(q).permute(1, 0, 2)
        sf0 = self.sf_embedding(sf0).permute(1, 0, 2)
        sf1 = self.sf_embedding(sf1).permute(1, 0, 2)
        sf2 = self.sf_embedding(sf2).permute(1, 0, 2)
        sf3 = self.sf_embedding(sf3).permute(1, 0, 2)
        sf4 = self.sf_embedding(sf4).permute(1, 0, 2)
        sf5 = self.sf_embedding(sf5).permute(1, 0, 2)
        sf6 = self.sf_embedding(sf6).permute(1, 0, 2)
        sf7 = self.sf_embedding(sf7).permute(1, 0, 2)

        _, (emb_q, _) = self.q_rnn(q)

        _, (emb_sf0, _) = self.sf_rnn(sf0)
        _, (emb_sf1, _) = self.sf_rnn(sf1)
        _, (emb_sf2, _) = self.sf_rnn(sf2)
        _, (emb_sf3, _) = self.sf_rnn(sf3)
        _, (emb_sf4, _) = self.sf_rnn(sf4)
        _, (emb_sf5, _) = self.sf_rnn(sf5)
        _, (emb_sf6, _) = self.sf_rnn(sf6)
        _, (emb_sf7, _) = self.sf_rnn(sf7)

        emb_sfs = [emb_sf0, emb_sf1, emb_sf2, emb_sf3, emb_sf4, emb_sf5, emb_sf6, emb_sf7]

        g_o = Variable(torch.zeros(32,256))

        if torch.cuda.is_available():
            g_o = g_o.cuda()

        for i, _ in enumerate(emb_sfs):
            o_i = emb_sfs[i]
            for j, _ in enumerate(emb_sfs):
                o_j = emb_sfs[j]
                #print(o_i.shape)
                #print(o_j.shape)
                #print(emb_q.shape)
                x = F.relu(self.g1(torch.cat((o_i,o_j,emb_q),2)))
                x = F.relu(self.g2(x))
                x = F.relu(self.g3(x))
                x = F.relu(self.g4(x))
                g_o.add(x)

        x = F.relu(self.f1(g_o))
        x = F.relu(self.f2(x))
        x = self.f3(x)

        return x
