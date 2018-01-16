# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from data import Corpus
import argparse
from model import RNNModel
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_path', default='./')
    parser.add_argument('-l', '--load_path', default=None)
    args = parser.parse_args()
    return args


cl_args = parse_args()
dataset = Corpus()
dataset.process_data()

args = {}
args['ntokens_source'] = len(dataset.source_dict)
args['ntokens_target'] = len(dataset.target_dict)
args['nhid'] = 128
args['em_size'] = 128
args['nlayers'] = 1
args['dropout'] = 0.0
args['rnn_type'] = 'LSTM'

args['batch_size'] = 20
args['n_epochs'] = 50
args['log_interval'] = 100
args['learning_rate'] = 20
np.save(os.path.join(cl_args.save_path, 'args.npy'), args)


if not os.path.exists(cl_args.save_path):
    os.makedirs(cl_args.save_path)

criterion = nn.CrossEntropyLoss(
    ignore_index=dataset.target_dict.word2idx['<pad>']
)

model = RNNModel(args).cuda()
optimizer = torch.optim.Adam(model.parameters())


if cl_args.load_path:
    save_file = os.path.join(cl_args.load_path, 'model.pt')
    model.load_state_dict(torch.load(save_file))


def loop(which_set, lr=None):
    if which_set is 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_acc = 0.
    total_length = 0

    start_time = time.time()
    itr = dataset.create_epoch_iterator(which_set, args['batch_size'])
    for i, (source, target) in enumerate(itr):
        output = model(source, target[:-1])
        output_flat = output.contiguous().view(-1, args['ntokens_target'])
        loss = criterion(output_flat, target[1:].view(-1))

        sample = F.softmax(output_flat, 1).max(1)[1].squeeze()
        acc = sample.eq(target[1:].view(-1))
        acc = acc.float().mean()

        total_loss += len(target) * loss.data
        total_length += len(target)
        total_acc += acc.data

        if which_set == 'train':
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if which_set == 'train' and i % args['log_interval'] == 0 and i > 0:
            cur_loss = total_loss[0] / total_length
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | acc {:5.2f}'.format(
                      epoch, i, len(
                          dataset.data[which_set]) // args['batch_size'], lr,
                      elapsed * 1000 / args['log_interval'], cur_loss, total_acc[0] / i))
            start_time = time.time()

    return total_loss[0] / total_length, total_acc[0] / i


lr = args['learning_rate']
best_val_loss = None
try:
    for epoch in range(1, args['n_epochs'] + 1):
        epoch_start_time = time.time()
        loop('train', lr)
        val_loss, val_acc = loop('valid')
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid acc {:8.2f}'.format(
                  epoch, (time.time() - epoch_start_time),
                  val_loss, val_acc)
                )
        print('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            file = os.path.join(cl_args.save_path, 'model.pt')
            torch.save(model.state_dict(), file)
            best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
file = os.path.join(cl_args.save_path, 'model.pt')
model.load_state_dict(torch.load(file))

# Run on test data.
test_loss, test_acc = loop('test')
print('=' * 89)
print('| End of training | test loss {:5.2f} | test acc {:8.2f}'.format(
    test_loss, test_acc))
print('=' * 89)
