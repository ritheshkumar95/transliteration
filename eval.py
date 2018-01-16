# coding: utf-8
import torch
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


def target_sent2idx(target):
    out = [dataset.target_dict.idx2word[x]
           for x in target.cpu().data.squeeze().numpy()]
    try:
        out = out[:out.index('<eos>')+1]
    except:
        pass
    return out


def source_sent2idx(target):
    return [dataset.source_dict.idx2word[x]
            for x in target.cpu().data.squeeze().numpy()]


model = RNNModel(args).cuda()
model.eval()
if cl_args.load_path:
    file = os.path.join(cl_args.load_path, 'model.pt')
    model.load_state_dict(torch.load(file))

itr = dataset.create_epoch_iterator('test', 1)
count = 0
total = 0
for i, (source, target) in enumerate(itr):
    total += 1
    output = model.sample(source, sos, eos)
    source = source_sent2idx(source)
    target = target_sent2idx(target)
    output = target_sent2idx(output)
    print "Source: ", ''.join(source)
    print "Original: ", ''.join(target)
    print "Generated: ", ''.join(output)
    print "\n"
    if len(target) == len(output) and (np.asarray(target) == np.asarray(output)).all():
        count += 1
print "{} / {} correct!".format(count, total)
