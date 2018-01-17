import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, cf):
        super(RNNModel, self).__init__()
        self.cf = cf
        self.drop = nn.Dropout(cf.dropout)
        self.encoder_emb = nn.Embedding(cf.ntokens_source, cf.em_size)
        self.decoder_emb = nn.Embedding(cf.ntokens_target, cf.em_size)
        self.encoder_rnn = getattr(nn, cf.rnn_type)(
            cf.em_size, cf.nhid, cf.nlayers, dropout=cf.dropout)
        self.decoder_rnn = getattr(nn, cf.rnn_type)(
            cf.em_size, cf.nhid, cf.nlayers, dropout=cf.dropout, bidirectional=True)
        self.decoder = nn.Linear(2*cf.nhid, cf.ntokens_target)

    def encode(self, source, source_lengths):
        emb = self.drop(self.encoder_emb(source))
        hidden = self.init_hidden(source.size(1))

        # source_lengths, perm_idx = torch.from_numpy(
        #     source_lengths
        # ).cuda().sort(0, descending=True)
        # emb = emb[:, perm_idx]
        # emb = pack_padded_sequence(emb, source_lengths.cpu().tolist())
        _, (h0, c0) = self.encoder_rnn(emb, hidden)

        h0 = h0[-1].expand(2 * h0.size(0), h0.size(1), h0.size(2)).contiguous()
        c0 = c0[-1].expand(2 * c0.size(0), c0.size(1), c0.size(2)).contiguous()
        h0[1::2].data.zero_()
        c0[1::2].data.zero_()
        hidden = (h0, c0)
        return hidden
    
    def decode(self, target, target_lengths, hidden):
        emb = self.drop(self.decoder_emb(target))

        emb = pack_padded_sequence(emb, target_lengths)
        output, _ = self.decoder_rnn(emb, hidden)
        output = pad_packed_sequence(output)[0]

        fwd, bwd = output.split(self.cf.nhid, -1)
        new_bwd = torch.zeros_like(bwd)
        new_bwd[:-2] = bwd[2:]
        output = torch.cat([fwd, new_bwd], -1)
        return self.decoder(self.drop(output))

    def forward(self, source, source_lengths, target, target_lengths):
        hidden = self.encode(source, source_lengths)
        out = self.decode(target, target_lengths, hidden)
        return out

    def sample(self, source, source_lengths, target, target_lengths, sos, eos):
        hidden = self.encode(source, source_lengths)
        # target = Variable(
        #     torch.from_numpy(
        #         np.random.randint(0, self.cf.ntokens_target, (target.size(0), len(target_lengths)))
        #     ).long()
        # ).cuda()
        # target[0].data.fill_(sos)
        # target[-1].data.fill_(eos)

        for j in tqdm(xrange(100)):
            for i in xrange(target.size(0)-2):
                output = self.decode(target, target_lengths, hidden)
                output = F.softmax(output[i], 1).max(1)[1].squeeze()
                # output = F.softmax(output[i], 1).multinomial(1).squeeze()
                target[i+1].data.copy_(output.data)
        return target

    def init_hidden(self, bsz, bidirec=False):
        weight = next(self.parameters()).data
        nlayers = self.cf.nlayers
        if bidirec:
            nlayers *= 2
        nhid = self.cf.nhid
        if self.cf.rnn_type == 'LSTM':
            return (Variable(weight.new(nlayers, bsz, nhid).zero_()),
                    Variable(weight.new(nlayers, bsz, nhid).zero_()))
        else:
            return Variable(weight.new(nlayers, bsz, nhid).zero_())
