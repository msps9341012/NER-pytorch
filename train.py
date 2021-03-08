# coding=utf-8
from __future__ import print_function
#import optparse
import itertools
#from collections import OrderedDict
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


t = time.time()

opts, parameters=get_args()
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

'''
word_sorted_items = sorted(dico_words.items(), key=lambda x: (-x[1], x[0]))
word_freq=list(map(lambda x: x[1],word_sorted_items))
word_freq[0]=0
word_freq[1]=0
word_freq=np.array(word_freq)/sum(word_freq)
'''

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
'''
char_sorted_items=sorted(dico_chars.items(), key=lambda x: (-x[1], x[0]))
char_freq=list(map(lambda x: x[1],char_sorted_items))
char_freq[0]=0
char_freq=np.array(char_freq)/sum(char_freq)
'''

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
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()

learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
losses = []
#loss = 0.0
best_dev_F = -1.0
best_test_F = -1.0
best_train_F = -1.0
all_F = [[0, 0, 0]]
plot_every = 10
eval_every = 20
count = 0

best_idx=0

#vis = visdom.Visdom()



sys.stdout.flush()


from conlleval import evaluate

def evaluating(model, datas, best_F,display_confusion_matrix = False):

    true_tags=[]
    pred_tags=[]

    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            true_tags.append(id_to_tag[true_id])
            pred_tags.append(id_to_tag[pred_id])
            confusion_matrix[true_id, pred_id] += 1
    
    prec, rec, new_F = evaluate(true_tags, pred_tags, verbose=False)

    if new_F > best_F:
        best_F = new_F
        save = True
        #print('the best F is ', new_F)

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

def extract_grad_hook(module, grad_in, grad_out):
    if module.weight.shape[0]==len(char_to_id): #char_level
        extracted_grads_char.append(grad_out[0])
    if module.weight.shape[0]==word_embeds.shape[0]: #word_level
        extracted_grads_word.append(grad_out[0])


# add hooks for embeddings, only add a hook to encoder wordpiece embeddings (not position)
def add_hooks(model):
    hook_registered = False
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.requires_grad = True
            module.register_backward_hook(extract_grad_hook)
            hook_registered = True
    if not hook_registered:
        exit("Embedding matrix not found")

if parameters['adv']:
    add_hooks(model)

#word_freq_scale=torch.tensor(word_freq, requires_grad=False).float().unsqueeze(1).cuda()
#char_freq_scale=torch.tensor(char_freq, requires_grad=False).float().unsqueeze(1).cuda()
'''
def normalize(freq_scale,emb):
    mean = (freq_scale * emb).sum(axis=0, keepdims=True) 
    var=(freq_scale * (emb - mean)**2.).sum(axis=0, keepdims=True)
    stddev = torch.sqrt(1e-6 + var)
    return (emb - mean) / stddev

if parameters['norm']:
    model.char_embeds.weight.data=normalize(char_freq_scale,model.char_embeds.weight.data)
    model.word_embeds.weight.data=normalize(word_freq_scale,model.word_embeds.weight.data)
'''

def gen_input(input_data):
    data = input_data
    sentence_in = Variable(torch.LongTensor(data['words'])) 
    targets = torch.LongTensor(data['tags'])                
    caps = Variable(torch.LongTensor(data['caps']))          
    chars2 = data['chars']    
    
    if parameters['char_mode'] == 'LSTM':
        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        matching_char = {}

        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj and not j in matching_char and not i in matching_char.values():
                    matching_char[j] = i
                    continue          
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)

        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))
        
    if parameters['char_mode'] == 'CNN':
        matching_char = {}
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))
    
    return  sentence_in, targets, chars2_mask, caps, chars2_length, matching_char


model.train(True)
ratio=0.0
if parameters['adv']:
    ratio=0.5

if parameters['paraphrase']:
    #paraphraser=Paraphraser('english-ewt-ud-2.5-191206.udpipe')
    with open('../para_text/para_token_agg', 'rb') as handle:
        para_token_map = pickle.load(handle)
    
    from weight_scheduler import WarmupWeight
    num_iters=parameters['epochs']*len(para_token_map.keys())
    warmup_iter=parameters['warmup']*num_iters
    
    ratio_scheduler=WarmupWeight(start_lr=0.5, warmup_iter=warmup_iter, num_iters=num_iters,
                              warmup_style=parameters['warmup_style'], last_iter=-1, alpha=parameters['exp_weight'])
    

for epoch in range(parameters['epochs']):
    in_epoch_losses = []
    for i, index in enumerate(np.random.permutation(len(train_data))):
        tr = time.time()
        count += 1

        extracted_grads_char = []
        extracted_grads_word = []

        data = train_data[index]
        model.zero_grad()
        sentence_in, targets, chars2_mask, caps, chars2_length, matching_char= gen_input(data)

        if use_gpu:
            sentence_in = sentence_in.cuda()
            targets     = targets.cuda()
            chars2_mask = chars2_mask.cuda()
            caps        = caps.cuda()

        neg_log_likelihood = model.neg_log_likelihood(sentence = sentence_in, 
                                                      tags = targets, 
                                                      chars2 = chars2_mask, 
                                                      caps = caps, 
                                                      chars2_length = chars2_length, 
                                                      matching_char = matching_char)
        
        neg_log_likelihood_para=0
        if parameters['paraphrase']:
            if index in para_token_map:
                
                ratio=ratio_scheduler.step()
                
                res = para_token_map[index]
                random_paraphrase = res[random.randint(0,len(res)-1)]
                
                sentence_in_para, targets_para, chars2_mask_para, caps_para, chars2_length_para, matching_char_para = gen_input(random_paraphrase)
                
                if use_gpu:
                    sentence_in_para = sentence_in_para.cuda()
                    targets_para     = targets_para.cuda()
                    chars2_mask_para = chars2_mask_para.cuda()
                    caps_para        = caps_para.cuda()
                
                neg_log_likelihood_para = model.neg_log_likelihood(sentence = sentence_in_para, 
                                                                  tags = targets_para, 
                                                                  chars2 = chars2_mask_para, 
                                                                  caps = caps_para, 
                                                                  chars2_length = chars2_length_para, 
                                                                  matching_char = matching_char_para)
                
            else:
                ratio = 0.0
                
        loss = float(neg_log_likelihood.cpu().detach().numpy()) / len(data['words'])
        neg_log_likelihood = neg_log_likelihood*(1-ratio)+ neg_log_likelihood_para*ratio
        neg_log_likelihood.backward()


        if parameters['adv']:
            neg_log_likelihood_adv = model.neg_log_likelihood(sentence = sentence_in, 
                                                          tags = targets, 
                                                          chars2 = chars2_mask, 
                                                          caps = caps, 
                                                          chars2_length = chars2_length, 
                                                          matching_char = matching_char,
                                                          adv=True,
                                                          grads=[extracted_grads_char[0]*2,extracted_grads_word[0]*2])


            neg_log_likelihood_adv = neg_log_likelihood_adv*(1-ratio)
            neg_log_likelihood_adv.backward()
            loss = loss + float(neg_log_likelihood_adv.cpu().detach().numpy()) / len(data['words'])


        in_epoch_losses.append(loss)
            
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()
        '''
        if parameters['norm']: 
            model.char_embeds.weight.data=normalize(char_freq_scale,model.char_embeds.weight.data)
            model.word_embeds.weight.data=normalize(word_freq_scale,model.word_embeds.weight.data)
        '''
        # if count % plot_every == 0:
        #     loss /= plot_every
        #     print(count, ': ', loss)
        #     if losses == []:
        #         losses.append(loss)
        #     losses.append(loss)
        #     text = '<p>' + '</p><p>'.join([str(l) for l in losses[-9:]]) + '</p>'
        #     losswin = 'loss_' + name
        #     textwin = 'loss_text_' + name
        #     vis.line(np.array(losses), X=np.array([plot_every*i for i in range(len(losses))]),
        #          win=losswin, opts={'title': losswin, 'legend': ['loss']})
        #     vis.text(text, win=textwin, opts={'title': textwin})
        #     loss = 0.0

        # if count % (eval_every) == 0 and count > (eval_every * 20) or \
        #         count % (eval_every*4) == 0 and count < (eval_every * 20):
        #     model.train(False)
        #     best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F)
        #     best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F)
        #     if save:
        #         torch.save(model, model_name)
        #     best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)
        #     sys.stdout.flush()

        #     all_F.append([new_train_F, new_dev_F, new_test_F])
        #     Fwin = 'F-score of {train, dev, test}_' + name
        #     vis.line(np.array(all_F), win=Fwin,
        #          X=np.array([eval_every*i for i in range(len(all_F))]),
        #          opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})
        #     model.train(True)

        # if count % len(train_data) == 0:
        #     adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))
    
    losses.append(np.mean(in_epoch_losses))
    model.train(False)


    #best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F)
    best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F)
    if save:
        best_idx = epoch
        torch.save(model.state_dict(), model_name)
    best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)

    all_F.append([0.0, new_dev_F, new_test_F])

    sys.stdout.flush()

    print('Epoch %d : valid/test/test_best : %.2f / %.2f / %.2f - %d'%(epoch, best_dev_F, new_test_F,best_test_F, best_idx))
    model.train(True)
    adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))



print(time.time() - t)

#plt.plot(losses)
#plt.show()
'''
max_temp=max_idx=0
for i in range(len(all_F)):
    if all_F[i][2] > max_temp:
        max_temp = all_F[i][2]
        max_idx = i
print(max_idx, max_temp)
'''