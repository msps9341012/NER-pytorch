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


experiment = Experiment(api_key='Bq7FWdV8LPx8HkWh67e5UmUPm',
                       project_name='NER',
                       auto_param_logging=False, auto_metric_logging=False)

experiment.log_parameters(parameters)


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

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    cPickle.dump(mappings, f)

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



model.train(True)
ratio=0.0
if parameters['adv']:
    ratio=0.5
    
    
if parameters['paraphrase']:
    #paraphraser=Paraphraser('english-ewt-ud-2.5-191206.udpipe')
    with open('../para_text/para_token_agg_10', 'rb') as handle:
        para_token_map = pickle.load(handle)
    
    from weight_scheduler import WarmupWeight
    num_iters=(parameters['epochs']-parameters['launch_epoch'])*len(para_token_map.keys())
    warmup_iter=parameters['warmup']*num_iters
    
    ratio_scheduler=WarmupWeight(start_lr=0.5, warmup_iter=warmup_iter, num_iters=num_iters,
                              warmup_style=parameters['warmup_style'], last_iter=-1, alpha=parameters['exp_weight'])
    

from conlleval import evaluate

def evaluating_batch(model, datas, best_F,display_confusion_matrix = False):

    true_tags=[]
    pred_tags=[]

    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
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



#random.shuffle(train_data)
if parameters['paraphrase']:
    '''
    since some examples don't have paraphrase, separate them into two parts.
    And then pack into batch.
    '''
    train_have_para=[]
    train_left=[]
    for i in range(len(train_data)):
        if i in para_token_map:
            train_have_para.append(train_data[i])
        else:
            train_left.append(train_data[i])
    
    train_have_para = generate_batch_data(train_have_para,5)
    train_left = generate_batch_data(train_left,5)
    para_batch = generate_batch_para(para_token_map,5)
    train_batched = train_have_para + train_left

elif parameters['word_rep']:
    '''
    same logic as above
    '''
    
    with open('../rep_text/rep_token_agg_complete', 'rb') as handle:
        rep_token_map = pickle.load(handle)
        
    train_have_rep=[]
    train_left=[]
    
    assert len(train_data)==len(rep_token_map), 'different length'
    for i in range(len(rep_token_map)):
        if rep_token_map[i]['str_words']:
            train_have_rep.append(train_data[i])            
        else:
            train_left.append(train_data[i])

    train_have_rep = generate_batch_data(train_have_rep,5)
    train_left = generate_batch_data(train_left,5)
    rep_batch = generate_batch_rep(rep_token_map,5)
    train_batched = train_have_rep + train_left
    
    from weight_scheduler import WarmupWeight
    #since rep_token_map may contain empty example, use rep_batch to approximate
    num_iters=(parameters['epochs']-parameters['launch_epoch'])*(len(rep_batch)*5)
    warmup_iter=parameters['warmup']*num_iters
    
    ratio_scheduler=WarmupWeight(start_lr=0.5, warmup_iter=warmup_iter, num_iters=num_iters,
                              warmup_style=parameters['warmup_style'], last_iter=-1, alpha=parameters['exp_weight'])
    
    
    
else:
    train_batched=generate_batch_data(train_data,5)

dev_batched=generate_batch_data(dev_data,1)
test_batched=generate_batch_data(test_data,1)




#To-do: combine para and rep together
       
metrics={}

for epoch in range(parameters['epochs']):
    
    in_epoch_losses = []
    in_epoch_losses_adv = []
    for i, index in enumerate(np.random.permutation(len(train_batched))):
        tr = time.time()
        count += 1

        extracted_grads_char = []
        extracted_grads_word = []

        #data = train_data[index]
        data=train_batched[index]
        model.zero_grad()
        #sentence_in, targets, chars2_mask, caps, chars2_length, matching_char= gen_input(data)
        
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

        neg_log_likelihood = model.loss(sentence = sentence_in, tags = targets, chars = chars2_mask, 
                                        caps = caps, chars2_length = chars2_length, word_length=word_length)
        
        
        #To-do: unified the loss variable name for para and rep.
        neg_log_likelihood_para=0
        if parameters['paraphrase']:
            if index < len(para_batch) and epoch >= parameters['launch_epoch']:
                
                ratio=ratio_scheduler.step()
                
                
                for para_data in para_batch[index]:
                    sentence_in_para=Variable(torch.LongTensor(para_data['words']))
                    chars2_mask_para=Variable(torch.LongTensor(para_data['chars']))
                    caps_para=Variable(torch.LongTensor(para_data['caps']))
                    targets_para = torch.LongTensor(para_data['tags']) 
                    chars2_length_para = para_data['char_length']
                    word_length_para=para_data['word_length']
                
                    if use_gpu:
                        sentence_in_para = sentence_in_para.cuda()
                        targets_para     = targets_para.cuda()
                        chars2_mask_para = chars2_mask_para.cuda()
                        caps_para        = caps_para.cuda()
                
                    neg_log_likelihood_para += model.loss(sentence = sentence_in_para, 
                                                         tags = targets_para, 
                                                         chars = chars2_mask_para, 
                                                         caps = caps_para, 
                                                         chars2_length = chars2_length_para, 
                                                         word_length=word_length_para)
                
                neg_log_likelihood_para=neg_log_likelihood_para/len(para_batch[index])
                in_epoch_losses_adv.append(float(neg_log_likelihood_para.cpu().detach().numpy()))
                
                
            else:
                ratio = 0.0
                
        neg_log_likelihood_rep=0
        if parameters['word_rep']:
            if index < len(rep_batch) and epoch >= parameters['launch_epoch']:
                
                ratio=ratio_scheduler.step()
                rep_data=rep_batch[index]
                
                sentence_in_rep=Variable(torch.LongTensor(rep_data['words']))
                chars2_mask_rep=Variable(torch.LongTensor(rep_data['chars']))
                caps_rep=Variable(torch.LongTensor(rep_data['caps']))
                targets_rep = torch.LongTensor(rep_data['tags']) 
                chars2_length_rep = rep_data['char_length']
                word_length_rep=rep_data['word_length']
                
                if use_gpu:
                    sentence_in_rep = sentence_in_rep.cuda()
                    targets_rep     = targets_rep.cuda()
                    chars2_mask_rep = chars2_mask_rep.cuda()
                    caps_rep        = caps_rep.cuda()

                neg_log_likelihood_rep = model.loss(sentence = sentence_in_rep, 
                                                     tags = targets_rep, 
                                                     chars = chars2_mask_rep, 
                                                     caps = caps_rep, 
                                                     chars2_length = chars2_length_rep, 
                                                     word_length=word_length_rep)
                
                in_epoch_losses_adv.append(float(neg_log_likelihood_rep.cpu().detach().numpy()))
                
                
            else:
                ratio = 0.0
                
        loss = float(neg_log_likelihood.cpu().detach().numpy())
        
        #To-do: unified the loss variable name for para and rep.
        neg_log_likelihood = neg_log_likelihood*(1-ratio)+ (neg_log_likelihood_para + neg_log_likelihood_rep)*ratio
        neg_log_likelihood.backward()
        
        in_epoch_losses.append(loss)
            
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()
        
        #if parameters['norm']: 
            #model.char_embeds.weight.data=normalize(char_freq_scale,model.char_embeds.weight.data)
            #model.word_embeds.weight.data=normalize(word_freq_scale,model.word_embeds.weight.data)
    losses.append(np.mean(in_epoch_losses))
    
    metrics['loss_norm']=np.mean(in_epoch_losses)
    if epoch >= parameters['launch_epoch'] and (parameters['paraphrase'] or parameters['word_rep']):
        metrics['loss_adv']=np.mean(in_epoch_losses_adv)
    else:
        metrics['loss_adv']=0
    
    
    model.train(False)
    best_dev_F, new_dev_F, save = evaluating_batch(model, dev_batched, best_dev_F)
    if save:
        best_idx = epoch
        torch.save(model.state_dict(), model_name)
    best_test_F, new_test_F, _ = evaluating_batch(model, test_batched, best_test_F)

    all_F.append([0.0, new_dev_F, new_test_F])
    
    
    sys.stdout.flush()
    print('Epoch %d : valid/test/test_best : %.2f / %.2f / %.2f - %d'%(epoch, best_dev_F, new_test_F,best_test_F, best_idx))
    model.train(True)
    adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))
    
    metrics['new_test_F']=new_test_F
    metrics['new_dev_F']=new_dev_F
    
    experiment.log_metrics(metrics)
    experiment.set_step(epoch+1)
    
    


print(time.time() - t)
