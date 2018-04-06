import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RelationNetwork(nn.Module):
    def __init__(self, q_vocab_size, sf_vocab_size, a_vocab_size):
        super().__init__()

        dropout = 0.25

        self.embedding = nn.Embedding(q_vocab_size, 32)

        self.q_rnn = nn.LSTM(32, 32, dropout=dropout)
        self.sf_rnn = nn.LSTM(32, 32, dropout=dropout)
        
        self.g1 = nn.Linear(32*2+1, 256)
        self.g2 = nn.Linear(256, 256)
        self.g3 = nn.Linear(256, 256)
        self.g4 = nn.Linear(256, 256)

        self.f1 = nn.Linear(256, 256)
        self.f2 = nn.Linear(256, 512)
        self.f3 = nn.Linear(512, a_vocab_size)

        self.do = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
            else:
                nn.init.constant(p, 0)

    def forward(self, q, sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7):

        q = self.do(self.embedding(q).permute(1, 0, 2))
        
        sf0 = self.do(self.embedding(sf0).permute(1, 0, 2))
        sf1 = self.do(self.embedding(sf1).permute(1, 0, 2))
        sf2 = self.do(self.embedding(sf2).permute(1, 0, 2))
        sf3 = self.do(self.embedding(sf3).permute(1, 0, 2))
        sf4 = self.do(self.embedding(sf4).permute(1, 0, 2))
        sf5 = self.do(self.embedding(sf5).permute(1, 0, 2))
        sf6 = self.do(self.embedding(sf6).permute(1, 0, 2))
        sf7 = self.do(self.embedding(sf7).permute(1, 0, 2))

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

        emb_q = emb_q.squeeze(0)

        for i, _ in enumerate(emb_sfs):
            o_i = emb_sfs[i].squeeze(0)
            pos = Variable(torch.FloatTensor(o_i.shape[0], 1).fill_(i))
            if torch.cuda.is_available():
                pos = pos.cuda()
            x = self.do(F.relu(self.g1(torch.cat((o_i,emb_q,pos),dim=1))))
            x = self.do(F.relu(self.g2(x)))
            x = self.do(F.relu(self.g3(x)))
            x = self.do(F.relu(self.g4(x)))
            g_o = g_o.add(x)

        x = self.do(F.relu(self.f1(g_o)))
        x = self.do(F.relu(self.f2(x)))
        x = self.f3(x)

        return x
