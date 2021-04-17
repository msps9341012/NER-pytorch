# coding=utf-8


from __future__ import print_function

from comet_ml import Experiment
from tqdm import tqdm
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
from pytorchtools import EarlyStopping




t = time.time()

opts, parameters=get_args()

experiment=None
'''
experiment = Experiment(api_key='Bq7FWdV8LPx8HkWh67e5UmUPm',
                       project_name='NER',
                       auto_param_logging=False, auto_metric_logging=False)

experiment.log_parameters(parameters)
'''

models_path = "models/"
use_gpu = parameters['use_gpu']

mapping_file = 'models/mapping.pkl'

name = parameters['name']
model_name = models_path + name #get_name(parameters)


if not os.path.exists(models_path):
    os.makedirs(models_path)
    

early_stopping = EarlyStopping(patience=20, verbose=True, path=model_name)   

    
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

# with open(mapping_file, 'wb') as f:
#     mappings = {
#         'word_to_id': word_to_id,
#         'tag_to_id': tag_to_id,
#         'char_to_id': char_to_id,
#         'parameters': parameters,
#         'word_embeds': word_embeds
#     }
#     cPickle.dump(mappings, f)

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
sample_count = 0

best_idx=0


        
if parameters['reload']:
    print('loading model:', parameters['reload'])
    checkpoint = torch.load(models_path+parameters['reload'])
    #model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    adjust_learning_rate(optimizer, lr=learning_rate)


sys.stdout.flush()


from conlleval import evaluate



model.train(True)
ratio=0.0
if parameters['adv']:
    ratio=0.5
    
    


    

from conlleval import evaluate

def evaluating_batch(model, datas, best_F,display_confusion_matrix = False):

    true_tags=[]
    pred_tags=[]

    save = False
    new_F = 0.0
    
    loss=[]
    
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        sentence_in=Variable(torch.LongTensor(data['words']))
        chars2_mask=Variable(torch.LongTensor(data['chars']))
        caps=Variable(torch.LongTensor(data['caps']))
        targets = torch.LongTensor(data['tags']) 
        chars2_length = data['char_length']
        word_length=data['word_length']
        

        if use_gpu:
            sentence_in = sentence_in.cuda()
            targets     = targets.cuda()
            chars2_mask = chars2_mask.cuda()
            caps        = caps.cuda()
        val, out = model(sentence= sentence_in, caps = caps,chars = chars2_mask, 
                         chars2_length= chars2_length,word_length=word_length)
        
        
        ground_truth_id = []
        index=0
        for length in data['word_length']:
            ground_truth_id.append(data['tags'][index,:length])
            index=index+1
        
        ground_truth_id = np.concatenate(ground_truth_id)
        
        predicted_id = np.concatenate(out)
        
        for (true_id, pred_id) in zip(ground_truth_id, predicted_id):
            true_tags.append(id_to_tag[true_id])
            pred_tags.append(id_to_tag[pred_id])
            confusion_matrix[true_id, pred_id] += 1
    prec, rec, new_F = evaluate(true_tags, pred_tags, verbose=False)
    
    if new_F > best_F:
        best_F = new_F
        save = True

    if display_confusion_matrix:
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            "ID", "NE", "Total",
            *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
        ))
        for i in range(confusion_matrix.size(0)):
            print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
                str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
                *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
                  ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
            ))
    return best_F, new_F, save





def forward_step(model, data, avg=True):
    sentence_in = Variable(torch.LongTensor(data['words']))
    chars2_mask = Variable(torch.LongTensor(data['chars']))
    caps = Variable(torch.LongTensor(data['caps']))
    targets = torch.LongTensor(data['tags']) 
    chars2_length = data['char_length']
    word_length=data['word_length']


    if use_gpu:
        sentence_in = sentence_in.cuda()
        targets     = targets.cuda()
        chars2_mask = chars2_mask.cuda()
        caps        = caps.cuda()

    return model.loss(sentence = sentence_in, tags = targets, chars = chars2_mask, caps = caps, chars2_length = chars2_length, word_length=word_length, avg=avg)







#random.shuffle(train_data)

#     train_have_para=[]
#     train_left=[]
#     for i in range(len(train_data)):
#         if i in para_token_map:
#             train_have_para.append(train_data[i])
#         else:
#             train_left.append(train_data[i])
    
#     train_have_para = generate_batch_data(train_have_para,5)
#     train_left = generate_batch_data(train_left,5)
#     para_batch = generate_batch_para(para_token_map,5)
#     train_batched = train_have_para + train_left

if parameters['non_gradient'] or parameters['dynamic_inference']:
    def divide_chunks(l, n):
        for i in range(0, len(l), n): 
            yield l[i:i + n]
            
    with open(parameters['adv_path'], 'rb') as handle:
        adv_data = pickle.load(handle)

    assert len(adv_data[0])==parameters['per_adv'], 'different number of adv_examples'
    assert len(train_data)==len(adv_data), 'different data length'
    
    if parameters['per_adv']==1:
        train_batched=generate_batch_data(train_data, parameters['batch_size'])
        adv_data = unpacked_data(adv_data)
        adv_data = prepare_dataset(adv_data, word_to_id, char_to_id, tag_to_id, lower)
        adv_batched = generate_batch_data(adv_data, parameters['batch_size'])
        
    else:
        train_have_adv=[]
        train_left=[]
        adv_examples=[]
        indexed_data_all = []
        
        number_of_adv=parameters['per_adv']
        print('processing adv data')
        with tqdm(total=len(adv_data)) as pbar:
            for index in range(len(adv_data)):
                if len(adv_data[index]) != number_of_adv:
                    train_left.append(train_data[index])

                else:
                    indexed_data = prepare_dataset(adv_data[index], word_to_id, char_to_id, tag_to_id, lower)

                    indexed_data_all.append(indexed_data)

                    adv_examples.append(generate_batch_data(indexed_data,number_of_adv)[0])
                    train_have_adv.append(train_data[index])
                
                pbar.update(1)
        
        train_have_adv = generate_batch_data(train_have_adv,parameters['batch_size'])
        
        train_left = generate_batch_data(train_left,parameters['batch_size'])
        train_batched = train_have_adv + train_left
        
        adv_batched = list(divide_chunks(adv_examples, parameters['batch_size']))
        indexed_data_batched = list(divide_chunks(indexed_data_all, parameters['batch_size']))
    
        assert len(adv_batched)==len(train_have_adv), 'different batch length'
    
    from weight_scheduler import WarmupWeight
    #since rep_token_map may contain empty example, use rep_batch to approximate
    num_iters=(parameters['epochs']-parameters['launch_epoch'])*len(adv_batched)
    warmup_iter=parameters['warmup']*num_iters
    
    ratio_scheduler=WarmupWeight(start_lr=0.5, warmup_iter=warmup_iter, num_iters=num_iters,
                              warmup_style=parameters['warmup_style'], last_iter=-1, alpha=parameters['exp_weight'])    
else:
    train_batched=generate_batch_data(train_data, parameters['batch_size'])


    
train_batched_ori=generate_batch_data(train_data,100)
dev_batched=generate_batch_data(dev_data, 100)
test_batched=generate_batch_data(test_data, 100)



def inference_and_filter(model, adv_example, index, pos):
    data_bank = indexed_data_batched[index][pos]
    with torch.no_grad():
        loss = forward_step(model, adv_example, avg=False)
        rank = torch.argsort(loss,descending=True)
        sel_index = rank[rank<5]
        sel_index = sel_index.cpu().detach().numpy()
    
    sel_example=[]
    for i in sel_index:
        sel_example.append(data_bank[i])
    
    return generate_batch_data(sel_example,5)[0]





       
metrics={}

disable_flag=not parameters['early_stop']

for epoch in range(parameters['epochs']):
    
    in_epoch_losses = []
    in_epoch_losses_adv = []
    for i, index in enumerate(np.random.permutation(len(train_batched))):
        tr = time.time()
        sample_count += 1

        extracted_grads_char = []
        extracted_grads_word = []

        #data = train_data[index]
        data=train_batched[index]
        model.zero_grad()
        #sentence_in, targets, chars2_mask, caps, chars2_length, matching_char= gen_input(data)
        
        neg_log_likelihood = forward_step(model, data)
        
        neg_log_likelihood_adv=0
        
        if parameters['dynamic_inference']:
            if index < len(adv_batched) and epoch >= parameters['launch_epoch']:
                
                ratio=ratio_scheduler.step()
                adv_batch_input = adv_batched[index]
                
                count=0
                for adv_example in adv_batch_input:
                    sel_data = inference_and_filter(model, adv_example, index, count)
                    
                    neg_log_likelihood_adv += forward_step(model, sel_data)
                    
                    count=count+1
                    
                neg_log_likelihood_adv=neg_log_likelihood_adv/count
                in_epoch_losses_adv.append(float(neg_log_likelihood_adv.cpu().detach().numpy()))


            else:
                ratio = 0.0
        
        if parameters['non_gradient']:
            if index < len(adv_batched) and epoch >= parameters['launch_epoch']:
                
                ratio=ratio_scheduler.step()
                
                adv_batch_input = adv_batched[index]
                
                if parameters['per_adv']==1:
                    adv_batch_input=[adv_batch_input]
                count=0

                for adv_example in adv_batch_input:
                    neg_log_likelihood_adv += forward_step(model, adv_example)
                    count=count+1
                    
                        
                neg_log_likelihood_adv=neg_log_likelihood_adv/count
                in_epoch_losses_adv.append(float(neg_log_likelihood_adv.cpu().detach().numpy()))
                
                
            else:
                ratio = 0.0
                
       
                
        loss = float(neg_log_likelihood.cpu().detach().numpy())
        
        neg_log_likelihood = neg_log_likelihood*(1-ratio)+ neg_log_likelihood_adv *ratio
        neg_log_likelihood.backward()
        
        in_epoch_losses.append(loss)
            
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()
        
        #if parameters['norm']: 
            #model.char_embeds.weight.data=normalize(char_freq_scale,model.char_embeds.weight.data)
            #model.word_embeds.weight.data=normalize(word_freq_scale,model.word_embeds.weight.data)
    losses.append(np.mean(in_epoch_losses))
    
#     metrics['loss_norm']=np.mean(in_epoch_losses)
#     if epoch >= parameters['launch_epoch'] and parameters['non_gradient']:
#         metrics['loss_adv']=np.mean(in_epoch_losses_adv)
#     else:
#         metrics['loss_adv']=0
    
    
    model.train(False)
    
    _, new_train_F, _ = evaluating_batch(model, train_batched_ori, 0)
    
    best_dev_F, new_dev_F, save = evaluating_batch(model, dev_batched, best_dev_F)

    
    if not disable_flag:
        if not early_stopping.early_stop:
            early_stopping(-new_dev_F, model, optimizer)
        else:
            print("Early stopping, now introduce adv examples")
            parameters['launch_epoch']=epoch
            disable_flag = 1 
            sample_count = len(train_batched)
            
    else:
        if save:
            torch.save(model.state_dict(), model_name)
            best_idx = epoch

    
        
    best_test_F, new_test_F, _ = evaluating_batch(model, test_batched, best_test_F)
    

    all_F.append([0.0, new_dev_F, new_test_F])
    
    
    sys.stdout.flush()
    print('Epoch %d : train/dev/test : %.2f / %.2f / %.2f - %d'%(epoch, new_train_F, new_dev_F, new_test_F, best_idx))
    model.train(True)
    adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*sample_count/len(train_data)))
    '''
    metrics['new_train_F']=new_train_F
    metrics['new_test_F']=new_test_F
    metrics['new_dev_F']=new_dev_F
    
    experiment.log_metrics(metrics)
    experiment.set_step(epoch+1)
    '''
    


print(time.time() - t)
