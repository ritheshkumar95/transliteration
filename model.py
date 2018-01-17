import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


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
            cf.em_size, cf.nhid, cf.nlayers, dropout=cf.dropout)
        self.decoder = nn.Linear(cf.nhid, cf.ntokens_target)

    def forward(self, source, lengths, target):
        emb = self.drop(self.encoder_emb(source))
        hidden = self.init_hidden(source.size(1))
        emb = pack_padded_sequence(emb, lengths)
        source, (h0, c0) = self.encoder_rnn(emb, hidden)
        h0 = h0[-1].expand(*h0.size()).contiguous()
        c0 = c0[-1].expand(*c0.size()).contiguous()
        hidden = (h0, c0)

        emb = self.drop(self.decoder_emb(target))
        output, _ = self.decoder_rnn(emb, hidden)
        return self.decoder(self.drop(output))

    def sample(self, source, lengths, sos, eos):
        emb = self.encoder_emb(source)
        hidden = self.init_hidden(source.size(1))
        emb = pack_padded_sequence(emb, lengths)
        source, (h0, c0) = self.encoder_rnn(emb, hidden)
        h0 = h0[-1].expand(*h0.size()).contiguous()
        c0 = c0[-1].expand(*c0.size()).contiguous()
        hidden = (h0, c0)

        target = Variable(torch.LongTensor(1, len(lengths)).cuda())
        target.data.fill_(sos)
        answer = []
        for i in xrange(50):
            answer.append(target.clone())
            emb = self.decoder_emb(target)
            output, hidden = self.decoder_rnn(emb, hidden)
            output = F.softmax(self.decoder(output)[0], 1).max(1)[1]
            target.data.copy_(output.data)
        return torch.cat(answer, 0)

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
