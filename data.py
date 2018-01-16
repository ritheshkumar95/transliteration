import torch
import numpy as np
from torch.autograd import Variable
import xml.etree.ElementTree as ET


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self):
        self.source_dict = Dictionary()
        self.target_dict = Dictionary()

    def process_data(self):
        files = {}
        files['train'] = 'data/Training/NEWS2012-Training-EnHi-13937.xml'
        files['test'] = 'data/Ref/NEWS2012-Ref-EnHi-1000.xml'

        self.data = {}
        self.source_dict.add_word('<pad>')
        self.target_dict.add_word('<pad>')
        sos = self.target_dict.add_word('<sos>')
        eos = self.target_dict.add_word('<eos>')
        for split in ['train', 'test']:
            tree = ET.parse(files[split])
            root = tree.getroot()
            children = list(root)
            tmp_data = []
            for child in children:
                source = list(child.find('SourceName').text.upper())
                target = list(child.find('TargetName').text)
                source = [self.source_dict.add_word(x) for x in source]
                target = [sos] + [self.target_dict.add_word(x) for x in target] + [eos]
                tmp_data.append([source, target])

            if split == 'train':
                np.random.shuffle(tmp_data)
                train_length = len(tmp_data)
                self.data['train'] = sorted(
                    tmp_data[:int(.8 * train_length)], key=lambda tup: len(tup[0]))
                self.data['valid'] = sorted(
                    tmp_data[int(.8 * train_length):], key=lambda tup: len(tup[0]))
            else:
                self.data[split] = sorted(tmp_data, key=lambda tup: len(tup[0]))

    def create_epoch_iterator(self, which_set, batch_size=16):
        data = self.data[which_set]
        for i in xrange(0, len(data), batch_size):
            batch_data = data[i: i + batch_size]
            source, target = zip(*batch_data)

            maxlen_source = max([len(x) for x in source])
            maxlen_target = max([len(x) for x in target])

            batch_source = np.full(
                (len(batch_data), maxlen_source),
                self.source_dict.word2idx['<pad>']
            )
            batch_target = np.full(
                (len(batch_data), maxlen_target),
                self.target_dict.word2idx['<pad>']
            )

            for i, sent in enumerate(list(source)):
                length = len(sent)
                batch_source[i, :length] = sent

            for i, sent in enumerate(list(target)):
                length = len(sent)
                batch_target[i, :length] = sent

            source = Variable(
                torch.from_numpy(batch_source).long()
            ).t().cuda()
            target = Variable(
                torch.from_numpy(batch_target).long()
            ).t().cuda()

            yield source, target


if __name__ == '__main__':
    loader = Corpus()
    loader.process_data()
    itr = loader.create_epoch_iterator('train')

    for i in xrange(10):
        source, target = itr.next()

    for i in xrange(source.size(1)):
        print "Source: ", ''.join(
            [loader.source_dict.idx2word[x] for x in source.cpu().data.numpy()[:, i]]
        )
        print "Target: ", ''.join(
            [loader.target_dict.idx2word[x] for x in target.cpu().data.numpy()[:, i]]
        )
