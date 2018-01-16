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
    parser.add_argument('-l', '--load_path', default=None)
    args = parser.parse_args()
    return args


cl_args = parse_args()
dataset = Corpus()
dataset.process_data()
sos = dataset.target_dict.word2idx['<sos>']
eos = dataset.target_dict.word2idx['<eos>']
args = np.load(os.path.join(cl_args.load_path, 'args.npy')).tolist()


model = RNNModel(args).cuda()
model.eval()
if cl_args.load_path:
    file = os.path.join(cl_args.load_path, 'model.pt')
    model.load_state_dict(torch.load(file))

itr = dataset.create_epoch_iterator('test', 1)
for i in xrange(50):
    source, target = itr.next()
    output = model.sample(source, sos, eos)

    print "Source: ", ''.join(
        [dataset.source_dict.idx2word[x]
            for x in source.cpu().data.numpy()[:, 0]]
    )

    print "Original: ", ''.join(
        [dataset.target_dict.idx2word[x]
            for x in target.cpu().data.numpy()[:, 0]]
    )
    print "Generated: ", ''.join(
        [dataset.target_dict.idx2word[x]
            for x in output.cpu().data.numpy()[:, 0]]
    )
    print "\n"
    raw_input()
