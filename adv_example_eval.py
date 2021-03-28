# coding=utf-8


from __future__ import print_function

from comet_ml import Experiment
#import optparse
import itertools
from collections import OrderedDict
import loader
import torch
import time
import _pickle as cPickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
import random
import pickle

#import visdom

from utils import *
from loader import *
from model import BiLSTM_CRF

from arguments import get_args
from processor import generate_batch_data, generate_batch_para, generate_batch_rep



t = time.time()

opts, parameters=get_args()

experiment=None



models_path = "models/"
use_gpu = parameters['use_gpu']

mapping_file = 'models/mapping.pkl'

name = parameters['name']
model_name = models_path + name #get_name(parameters)
tmp_model = model_name + '.tmp'

if not os.path.exists(models_path):
    os.makedirs(models_path)
    
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)
test_train_sentences = loader.load_sentences(opts.test_train, lower, zeros)

update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)
update_tag_scheme(test_train_sentences, tag_scheme)

dico_words_train = word_mapping(train_sentences, lower)[0]

dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )



dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)


train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_train_data = prepare_dataset(
    test_train_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

all_word_embeds = {}
for i, line in enumerate(codecs.open(opts.pre_emb, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), opts.word_dim))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))


print('word_to_id: ', len(word_to_id))
model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'],
                   char_embedding_dim=parameters['char_dim'],
                   char_lstm_dim=parameters['char_lstm_dim'],
                   alpha=parameters['alpha'])
                   # n_cap=4,
                   # cap_embedding_dim=10)
        
if parameters['reload']:
    print('loading model')
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()

learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
losses = []
best_dev_F = -1.0
best_test_F = -1.0
best_train_F = -1.0
all_F = [[0, 0, 0]]
plot_every = 10
eval_every = 20
count = 0

best_idx=0


sys.stdout.flush()


from conlleval import evaluate



from conlleval import evaluate

import pandas as pd

def evaluating_batch(model, datas):

    save = False
    adv = 0
    true_tags_all=[]
    pred_tags_all=[]
    macro=[]
    for data in datas:
        
        true_tags = []
        pred_tags = []
        
        sentence_in=Variable(torch.LongTensor(data['words']))
        chars2_mask=Variable(torch.LongTensor(data['chars']))
        caps=Variable(torch.LongTensor(data['caps']))
        targets = torch.LongTensor(data['tags']) 
        chars2_length = data['char_length']
        word_length=data['word_length']
        ground_truth_id = data['tags'][0]
        
        if use_gpu:
            sentence_in = sentence_in.cuda()
            targets     = targets.cuda()
            chars2_mask = chars2_mask.cuda()
            caps        = caps.cuda()
        val, out = model(sentence= sentence_in, caps = caps,chars = chars2_mask, 
                         chars2_length= chars2_length,word_length=word_length)
        
        predicted_id = out
        
        for (true_id, pred_id) in zip(ground_truth_id, predicted_id[0]):
            true_tags.append(id_to_tag[true_id])
            pred_tags.append(id_to_tag[pred_id])
            
        
        true_tags_all.extend(true_tags)
        pred_tags_all.extend(pred_tags)
        
        df = pd.DataFrame({'true':true_tags,'pred':pred_tags})
        
        if sum(df['true']!=df['pred'])>0:
            adv = adv+1
        df=df[df['true']!='O'] #only tags
        if len(df)!=0:
            macro.append(sum(df['true']==df['pred'])/len(df))
        
    
    df_tags = pd.DataFrame({'true':true_tags_all,'pred':pred_tags_all})
    df_tags = df_tags[df_tags['true']!='O']
    
    
    print('Micro acc_tag:', sum(df_tags['true']==df_tags['pred'])/len(df_tags))
    print('Macro acc_tag:', np.mean(macro))
    prec, rec, new_F = evaluate(true_tags_all, pred_tags_all, verbose=False)
    print('F 1:',new_F)
    print('Hit:', adv/len(datas))



'''
should be raw data (not indexing one)
[[token,_, _,tag],...]
'''
adv_data_path='../para_text/para_ppdb_dev_all'
with open(adv_data_path, 'rb') as handle:
    adv_data = pickle.load(handle)


# res=[]
# for i in list(adv_data.values()):
#     res.append(i[0])
model.eval()
adv_data=prepare_dataset(adv_data, word_to_id, char_to_id, tag_to_id, lower)
adv_data=generate_batch_data(adv_data,1)
evaluating_batch(model, adv_data)
#
'''
dev_batched=generate_batch_data(dev_data,1)
evaluating_batch(model, dev_batched)
'''
