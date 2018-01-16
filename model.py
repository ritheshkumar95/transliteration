import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import ipdb


class RNNModel(nn.Module):
    def __init__(self, args):
        super(RNNModel, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args['dropout'])
        self.encoder_emb = nn.Embedding(args['ntokens_source'], args['em_size'])
        self.decoder_emb = nn.Embedding(args['ntokens_target'], args['em_size'])
        self.encoder_rnn = getattr(nn, args['rnn_type'])(
            args['em_size'], args['nhid'], args['nlayers'], dropout=args['dropout'])
        self.decoder_rnn = getattr(nn, args['rnn_type'])(
            args['em_size'], args['nhid'], args['nlayers'], dropout=args['dropout'])
        self.decoder = nn.Linear(args['nhid'], args['ntokens_target'])

    def forward(self, source, target):
        emb = self.drop(self.encoder_emb(source))
        hidden = self.init_hidden(source.size(1))
        source, hidden = self.encoder_rnn(emb, hidden)

        emb = self.drop(self.decoder_emb(target))
        output, _ = self.decoder_rnn(emb, hidden)
        return self.decoder(self.drop(output))

    def sample(self, source, sos, eos):
        emb = self.encoder_emb(source)
        hidden = self.init_hidden(source.size(1))
        source, hidden = self.encoder_rnn(emb, hidden)

        target = Variable(torch.LongTensor(1, source.size(1)).cuda())
        target.data.fill_(sos)
        answer = []
        for i in xrange(50):
            answer.append(target.clone())
            emb = self.decoder_emb(target)
            output, hidden = self.decoder_rnn(emb, hidden)
            output = F.softmax(self.decoder(output)[0], 1).max(1)[1]
            target.data.copy_(output.data)
        return torch.cat(answer, 0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        nlayers = self.args['nlayers']
        nhid = self.args['nhid']
        if self.args['rnn_type'] == 'LSTM':
            return (Variable(weight.new(nlayers, bsz, nhid).zero_()),
                    Variable(weight.new(nlayers, bsz, nhid).zero_()))
        else:
            return Variable(weight.new(nlayers, bsz, nhid).zero_())
