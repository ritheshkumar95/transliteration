# coding: utf-8
import torch
from config import load_config
import os
from data import Corpus
import argparse
from model import RNNModel
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-l', '--load_path', default=None)
    args = parser.parse_args()
    return args


args = parse_args()
dataset = Corpus()
dataset.process_data()
cf = load_config(args.path)
cf.ntokens_source = len(dataset.source_dict)
cf.ntokens_target = len(dataset.target_dict)
sos = dataset.target_dict.word2idx['<sos>']
eos = dataset.target_dict.word2idx['<eos>']


model = RNNModel(cf).cuda()
model.eval()
if args.load_path:
    model.load_state_dict(
        torch.load(os.path.join(args.load_path, 'model.pt'))
    )

itr = dataset.create_epoch_iterator('test', 16)
for i in xrange(50):
    source, source_lengths, target, target_lengths = itr.next()
    output = target.clone()
    output = model.sample(source, source_lengths, output, target_lengths, sos, eos)

    for j in xrange(source.size(1)):
        print "Source: ", ''.join(
            [dataset.source_dict.idx2word[x]
                for x in source.cpu().data.numpy()[:, j]]
        )

        print "Original: ", ''.join(
            [dataset.target_dict.idx2word[x]
                for x in target.cpu().data.numpy()[:, j]]
        )
        print "Generated: ", ''.join(
            [dataset.target_dict.idx2word[x]
                for x in output.cpu().data.numpy()[:, j]]
        )
        print "\n"
        raw_input()
