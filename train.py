import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import math, copy, time
from torch.autograd import Variable
from torch.utils.data import Dataset

import re
from itertools import groupby
import pickle
from utils import get_first_file
from torchtext.data import Iterator, BucketIterator

#### Device choice
num_device = 0
device = torch.device("cuda:" + str(num_device) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(num_device)
FLAGS = None

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.trg = trg
        
def data_gen_class(data_iter, TEXT, nbatches = None):
    iterator = iter(data_iter)
    len_max = len(data_iter)
    if nbatches == None:
        nbatches = len_max
    elif nbatches > len_max:
        nbatches = len_max
    for i in range(nbatches):
        bb = next(iterator)
        yield Batch(bb.Text[0].permute(1,0), bb.Label, TEXT.vocab.stoi['<pad>'])
        
        # Convolutional neural network
        
    
class ConvNet(nn.Module):
    def __init__(self, input_size, embed_size, num_classes=10):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)        
        self.conv1dBlock1 = nn.Sequential(
            nn.Conv1d(embed_size, 20, 3, padding=1),
            nn.Dropout(FLAGS.dropout),
            nn.BatchNorm1d(20),
            nn.ReLU())
        
        self.conv1dBlock2 = nn.Sequential(
            nn.Conv1d(20, 2, 3, padding=1),
            nn.Dropout(FLAGS.dropout),
            nn.BatchNorm1d(2),
            nn.ReLU())
        
    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1,2)
        x = self.conv1dBlock1(x)
        x = self.conv1dBlock2(x)
        out = torch.mean(x, dim=2)
        return out

def clean_line(line):
    hashtag = re.compile('#[^ \t\n\r\f\v]+')
    username = re.compile('@[^ \t\n\r\f\v]+')
    url = re.compile('https?[^ \t\n\r\f\v]*')
    junk = re.compile('["¤#%&()*+-/;<=>@[\]^_`{|}~\\;\(\)\'\"\*\`\´\‘\’…\\\/\{\}\|\+><~\[\]\“\”%=\$§]')
    ponctu = re.compile('[.!?,:]')
    number = re.compile('(^[0-9]+)|([0-9]+)')
    rep = re.compile(r'(.)\1{2,}')
    emo = re.compile('[\u233a-\U0001f9ff]')

    if line.startswith('"') :
        line = line[1:]
    if line.endswith('"') :
        line = line[:-1]
        
    line = re.sub(url,' url ', line) # replace every url with ' url '
    
    def subfct1(matchobj):
        return ' ' + matchobj.group(0) + ' '
    line = re.sub(ponctu,subfct1, line) # separate the punctuation from the words
    
    def subfct2(matchobj):
        return matchobj.group(0)[:2]
    line = re.sub(rep, subfct2,line) # keep maximum 2 consecutive identical character
    
    line = re.sub(hashtag,' hastag ', line) # replace every hastag with ' hastag '
    line = re.sub(username,' username ', line) # replace every reference to a username with ' username '
    line = re.sub(junk,' ', line) #throw away junk character
    line = re.sub(number,' number ',line) # replace every number with ' number '
    line = re.sub(emo, ' ',line) #suppr strange emoticon ( to modify ?)

    line_split = [k for k,v in groupby(line.split())] #suprr repeated word:
    line_split = line_split[:40] #trunc if too long
    return line_split

def custom_tokenizer_text(text): # create a tokenizer function
    return clean_line(text)

def custom_preprocess_label(label):
    label = int(label)
    if label == 4:
        label = 1
    return str(label)

def run_epoch(data_iter, model, criterion, TEXT, optimizer=None):
    nb_batches = len(data_iter)
    generator = data_gen_class(data_iter,TEXT)
    nb_item = 0
    total_loss = 0
    acc_y_sum = 0
    
    temp_nb_item = 0
    temp_total_loss = 0
    temp_acc_y_sum = 0
    
    for i, batch in enumerate(generator):
        #Forward pass
        outputs = model.forward(batch.src)
        loss = criterion(outputs, batch.trg)
        
        total_loss += loss.item()
        temp_total_loss += loss.item()
        acc_y = torch.argmax(outputs,dim=1) == batch.trg
        acc_y = acc_y.float().sum().item()
        acc_y_sum += acc_y
        temp_acc_y_sum += acc_y
        nb_item  += batch.src.size(0)
        temp_nb_item += batch.src.size(0)
        
        if optimizer is not None:
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 1000 == 0:
                print ('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}' .format(i+1, nb_batches, temp_total_loss/temp_nb_item, temp_acc_y_sum/temp_nb_item))
                temp_acc_y_sum = 0
                temp_nb_item = 0
                temp_total_loss = 0
    
    return total_loss/nb_item, acc_y_sum/nb_item

def train():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR','/stockage/Research_Team_Ressources/valohai_test/')#,'/valohai/inputs/')
    dataset_path = get_first_file(os.path.join(INPUTS_DIR, 'dataset'))
    word_vectors_path = get_first_file((os.path.join(INPUTS_DIR, 'word_vectors')))
                                       
    try:
        with open(word_vectors_path, 'rb') as my_pickle:
            TEXT = pickle.load(my_pickle)
    except IOError:
        print("IOError")
        pass

    LABEL = data.Field(sequential=False, preprocessing=custom_preprocess_label, use_vocab=False)                 
    dataset = data.TabularDataset(
        path = dataset_path, format='csv',
        fields=[('Num', None),('Label', LABEL), ('id', None), ('date',None),
                ('flag', None),('user', None),('Text', TEXT)],
        skip_header = True)
                                                   
    nb_train = 1000000
    ratio_train = nb_train / len(dataset)
    nb_test = 500000
    ratio_test = nb_test / len(dataset)
    ratio_other = 1 - ratio_train - ratio_test
    
    train_dataset, other_dataset, test_dataset = dataset.split(split_ratio=[ratio_train,ratio_test,ratio_other])
    
    train_iter, test_iter = BucketIterator.splits(
       (train_dataset, test_dataset), # we pass in the datasets we want the iterator to draw data from
       batch_sizes=(FLAGS.batch_size, FLAGS.batch_size),
       device=num_device, # if you want to use the GPU, specify the GPU number here
       sort_key=lambda x: len(x.Text), # the BucketIterator needs to be told what function it should use to group the data.
       sort_within_batch=False,
       repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
    )             
    
    n_vocab = len(TEXT.vocab)
    model = ConvNet(n_vocab, embed_size=FLAGS.embedding_size, num_classes = 2).to(device)
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
   
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion.to(device)
    
    num_epoch = FLAGS.epochs
    for epoch in range(num_epoch):
        print("epoch : ", epoch)
        model.train()
        print(run_epoch(train_iter, model,criterion, TEXT, optimizer))
        model.eval()
        print(run_epoch(test_iter, model,criterion, TEXT, None))
    
    model.eval()
    print(run_epoch(test_iter, model, criterion,TEXT, None))
    
    # Saving weights and biases as outputs of the task.
    outputs_dir = os.getenv('VH_OUTPUTS_DIR', '/valohai/outputs/')
    for i, ws in enumerate(all_weights):
        filename = os.path.join(outputs_dir, 'mytraining.pt')
        model.save_state_dict(filename)
    for i, bs in enumerate(all_biases):
        filename_text = os.path.join(outputs_dir, 'text.pickle')
        pickle.dump(TEXT,filename_text)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of steps to run trainer')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of elements in each batch')
    parser.add_argument('--embedding_size', type=int, default=300,
                        help='Size of embedding word vectors')
    FLAGS, unparsed = parser.parse_known_args()
    print("number of epochs : " + str(FLAGS.epochs))
    print("learning_rate : " + str(FLAGS.learning_rate))
    print("dropout : " + str(FLAGS.dropout))
    print("batch_size : " + str(FLAGS.batch_size))
    print("embedding_size : " + str(FLAGS.embedding_size))
    train()
    
