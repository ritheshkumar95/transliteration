# coding: utf-8
import torch
import os
from data import Corpus
import argparse
from model import RNNModel
import numpy as np
from config import load_config


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


model = RNNModel(cf).cuda()
model.eval()
if args.load_path:
    model.load_state_dict(
        torch.load(os.path.join(args.load_path, 'model.pt'))
    )

itr = dataset.create_epoch_iterator('test', 32)
target_list = []
generated_list = []
source_list = []
for i, (source, lengths, target) in enumerate(itr):
    output = model.sample(source, lengths, sos, eos)
    source_list += source.split(1, 1)
    target_list += target.split(1, 1)
    generated_list += output.split(1, 1)

count = 0
for source, target, output in zip(source_list, target_list, generated_list):
    source = source_sent2idx(source)
    target = target_sent2idx(target)
    output = target_sent2idx(output)
    print "Source: ", ''.join(source)
    print "Original: ", ''.join(target)
    print "Generated: ", ''.join(output)
    if len(target) == len(output) and (np.asarray(target) == np.asarray(output)).all():
        count += 1
        print "Correct!"
    print "\n"
print "{} / {} correct!".format(count, len(source_list))
