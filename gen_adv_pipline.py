import optparse
import os
import loader
from utils import *
from loader import *
from word_rep import Word_Replacement
import re
from tqdm import tqdm
import pickle
from paraphrase_model import Paraphraser
from ppdb import PPDB_Replacement
import torch
import sys
import numpy as np
from operator import itemgetter
from perplexity_est import Perplexity_estimator

lower = 1
zeros = 0
tag_scheme = 'iobes'
word_dim = 100


def add_args(parser):
    parser.add_option('--order', default='ppdb', help='the pipeline')
    parser.add_option("--n", default="1",type='int', help="number of adv examples to generate")
    parser.add_option("-p", "--pre_emb", default="models/glove.6B.100d.txt",help="Location of pretrained embeddings")
    parser.add_option('--dataset', default='train', help='using which dataset')
    parser.add_option('--wordbank', default='train', help='wordbank for word rep')
    parser.add_option("--preprocess_set", default="",help="Location of preprocessed set")
    parser.add_option("--name", default="",help="filename for saving")
    parser.add_option("--save_dir", default="",help="directory for storing the generated data")
    parser.add_option("--bert", action='store_true',help="using bert embedding")
    parser.add_option("--append_yago", action='store_true',help="append yago words in replacements")

    parser.add_option("--bert_pooler", default="mean",help="which bert pooling method")
    parser.add_option("--filter", action='store_true',help="apply perplexity filter or not")
    parser.add_option("--cascade", action='store_true',help="using the first example from the last step to generate")
    #To-do: add normal sampling
    parser.add_option("--rep_with", default="farthest",help="replace with closest|farthest embedding ")
    
    
    return parser


def load_data(pre_emb):

    train_sentences = loader.load_sentences('dataset/eng.train', lower, zeros)
    dev_sentences = loader.load_sentences("dataset/eng.testa", lower, zeros)
    test_sentences = loader.load_sentences("dataset/eng.testb", lower, zeros)

    update_tag_scheme(train_sentences, tag_scheme)
    update_tag_scheme(dev_sentences, tag_scheme)
    update_tag_scheme(test_sentences, tag_scheme)


    all_emb=1

    dico_words_train = word_mapping(train_sentences, lower)[0]
       
  
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        pre_emb,
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not all_emb else None
    )

    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
    

    all_word_embeds = {}
    for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == word_dim + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

    for w in word_to_id:
        if w in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

    print('Loaded %i pretrained embeddings.' % len(all_word_embeds))
    
    pretrained_word_list = set([line.rstrip().split()[0].strip() for line in codecs.open(pre_emb, 'r', 'utf-8') 
                            if len(pre_emb) > 0])
    
    
    filter_ppdb_list=[]
    for i in pretrained_word_list:
        if re.match('[^\x00-\x7F]+', i):
            continue
        else:
            if all(w in dico_chars for w in i):
                filter_ppdb_list.append(i)
    
    filter_ppdb_list = set(filter_ppdb_list) | set(dico_words_train.keys()) 
       
    train_sentences_packed = packed_data(train_sentences)
    dev_sentences_packed = packed_data(dev_sentences)
    test_sentences_packed = packed_data(test_sentences)
    
    return train_sentences_packed, dev_sentences_packed, test_sentences_packed, word_embeds, word_to_id, filter_ppdb_list

def using_word_rep(dataset, n, word_to_id, word_embeds, word_bank, rep_method, append_yago = False):
    '''
    If the data has already been processed via different method, then modify on them.
    Or, generate n adv examples using this method.
    '''


    if append_yago and not isinstance(word_bank, str):
        with open('../YagoReplacement/yago_wb.pkl', 'rb') as handle:
            word_bank_yago = pickle.load(handle)
        word_bank.extend(word_bank_yago)

    wr = Word_Replacement(lower, word_to_id, word_embeds, word_bank)
    
#     word_bank=unpacked_data(word_bank)
#     for sentence in word_bank:
#         wr.create_tag_chunks(sentence)

    print("Generating {} Adversarial Examples".format(rep_method))
    all_adversarial_examples_farthest = []

    print("Dataset Len : {}".format(len(dataset)))
    for sentence_pack in tqdm(dataset):
        if len(sentence_pack)==n:
            adv_example=[]
            for sentence in sentence_pack:
                adversarial_examples = wr.create_adv_examples(sentence, 1, "mean", rep_method)
                adv_example = adv_example + adversarial_examples
        else:
            adv_example = wr.create_adv_examples(sentence_pack[0], n, "mean", rep_method)

        all_adversarial_examples_farthest.append(adv_example)
    return all_adversarial_examples_farthest


def using_para(dataset, n):
    '''
    Since this method may not be able to generate n adv examples, be sure to put it in the last order.
    '''
    counter=0
    paraphraser=Paraphraser('english-ewt-ud-2.5-191206.udpipe',285)
    res=[]
    
    for sentence_pack in tqdm(dataset):
        adv_example=[]
        for sentence in sentence_pack:
            para_example=paraphraser.generate_n_paraphrases(sentence, 1)
            if para_example:
                counter=counter+1
                adv_example.append(para_example[0])
            else:
                adv_example.append(sentence)
        res.append(adv_example)
    print('%Examples modified by para:', counter/(len(res)*n))
    return res

def using_ppdb(dataset,n, word_map):
    '''
    If the data has already been processed via different method, then modify on them individually.
    Or, generate n adv examples using this method.
    '''
    ppdb = PPDB_Replacement('../ppdb/ppdb_all', word_map)
    res=[]
    index=0
    for sentence_pack in tqdm(dataset):
        adv_example=[]
        if len(sentence_pack)==n:
            for sentence in sentence_pack:
                adv_example.append(ppdb.para_examples(sentence,index))
        else:
            for _ in range(n):
                adv_example.append(ppdb.para_examples(sentence_pack[0],index))
        res.append(adv_example)
        index=index+1
    print('%Examples modified by ppdb:', len(ppdb.id_of_change_examples)/len(res))
    return res


def convert_to_string(ent):
    string_list = [i[0] for i in ent]
    string_list = " ".join(string_list).lower()
    string_list = re.sub(r' ([^A-Za-z0-9])', r'\1', string_list)
    return string_list
    

def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]
    
           
    
def filter_examples(sentence,adv_examples, n):
    print('filtering by perplexity score...')
    count=0
    filter_examples=[]
    sentence = unpacked_data(sentence)
    for sent, adv_list in tqdm(zip(sentence, adv_examples), total=len(sentence)):
        
        true_score = perplexity_filter.get_perplexity_score([convert_to_string(sent)],use_batch=False)[0]
        #perplexity_filter.perplexity_score(convert_to_string(sent))
        adv_string_list = list(map(convert_to_string, adv_list))
        
        
        #split into batch to prevent memory issue
        adv_string_batched = list(divide_chunks(adv_string_list, 20))
        
        score_list =[]
        for batch in adv_string_batched:
            score_list.append(perplexity_filter.get_perplexity_score(batch,use_batch=True))
        
        score_list = np.concatenate(score_list)
        
        sel_index = np.where(score_list<3*true_score)[0].tolist()
        
        sel_index = sel_index[:n]
        if len(sel_index)!=n:
            left_choice= list(set(list(range(100)))-set(sel_index))
            sel_index = sel_index + left_choice[:10-len(sel_index)]
            count=count+1
        
        assert len(sel_index)==n, 'filter error'
        filter_examples.append(list(itemgetter(*sel_index)(adv_list)))
    
    print('Fail to meet the threshold :',count)
        
    return filter_examples



def savefile(adv_data, opts, method):
    save_path=os.path.join(opts.save_dir, opts.name +"_"+ method[:-1])
    with open(save_path, 'wb') as handle:
        pickle.dump(adv_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_preprocessed(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data
    
    

def main():
            
    optparser = optparse.OptionParser()
    
    optparser = add_args(optparser)
    opts = optparser.parse_args()[0]
    
    number_to_generate = opts.n
    
    method_to_path={}
    pipeline_order = opts.order.split(',')

    append_yago = opts.append_yago
    
    if opts.preprocess_set:
        preprocess_set = opts.preprocess_set.split(',')
        preprocess_set = list(map(lambda x: x.strip(), preprocess_set))
        for filename in preprocess_set:
            if filename[-4:] in ['para', 'ppdb']:
                method_to_path[filename[-4:]] = os.path.join(opts.save_dir, filename)
            else:
                method_to_path['rep'] = os.path.join(opts.save_dir, filename)
    
    adv_by_para = None
    adv_by_ppdb = None
    adv_by_rep  = None
    
    if opts.filter:
        global_gen_number = 100
    else:
        global_gen_number = number_to_generate
    
    '''
    The data is packed 
    format is like [[example_1]]
    example_1: [[word,_,_,tag]...]
    ----
    The output data is also packed in the above format.
    '''  
    train_sentences, dev_sentences, test_sentences, word_embeds, word_to_id, filter_ppdb_list = load_data(opts.pre_emb)
    print('Finish loading data')

    
    dataset_map={'train':train_sentences, 'dev':dev_sentences, 'test':test_sentences}
    
    entry_data  = None
    updated_data = None
    agg_name = ""
    
    for method in pipeline_order:
        
        agg_name = agg_name + method + "_"
        
        if method=='ppdb':
            print('generate adv-examples via ppdb')
        
            if 'ppdb' in method_to_path:
                updated_data = load_preprocessed(method_to_path['ppdb'])
                print('used pre-processed data {}'.format(method_to_path['ppdb']))
            else:
                if updated_data:
                    print('used last step data')
                    data_to_ppdb=updated_data
                else:
                    print('used {}'.format(opts.dataset))
                    data_to_ppdb = dataset_map[opts.dataset]

                updated_data = using_ppdb(data_to_ppdb, global_gen_number, filter_ppdb_list)
                assert len(updated_data)==len(data_to_ppdb), 'error'
                if opts.filter:
                      updated_data = filter_examples(dataset_map[opts.dataset],updated_data, number_to_generate)
                savefile(updated_data, opts, agg_name)
            print('ppdb finished')
        
        if method=='para':
            print('generate adv-examples via para')
            if 'para' in method_to_path:
                updated_data = load_preprocessed(method_to_path['para'])
                print('used pre-processed data {}'.format(method_to_path['para']))
            else:
                if updated_data:
                    print('used last step data')
                    data_to_para=updated_data
                else:
                    print('used {}'.format(opts.dataset))
                    data_to_para = dataset_map[opts.dataset]
                    
                updated_data = using_para(data_to_para, global_gen_number)
                assert len(updated_data)==len(data_to_para), 'error'
                savefile(updated_data, opts, agg_name)
            print('para finished')
        
        if method=='rep':
        
            print('generate adv-examples via rep')

            if 'rep' in method_to_path:
                updated_data = load_preprocessed(method_to_path['rep'])
                print('used pre-processed data {}'.format(method_to_path['rep']))
            else:
                if updated_data:
                    print('used last step data')
                    data_to_rep=updated_data
                else:
                    print('used {}'.format(opts.dataset))
                    data_to_rep = dataset_map[opts.dataset]
                
                if opts.bert:
   

                    path_map = {'train':'../tag_embed/train_bert_{}'.format(opts.bert_pooler), 
                                'dev':'../tag_embed/dev_bert_{}'.format(opts.bert_pooler),
                                'dev_yago':'../tag_embed/dev_yago_bert_{}'.format(opts.bert_pooler),
                                'train_yago':'../tag_embed/train_yago_bert_{}'.format(opts.bert_pooler)}
                    
                    if append_yago:
                        word_bank_key = opts.wordbank + "_yago"
                    
                    word_bank_path = path_map[word_bank_key]
                    
                    updated_data = using_word_rep(data_to_rep, global_gen_number,  
                                                  word_to_id, 'bert', word_bank_path, opts.rep_with, opts.append_yago)
                else:
                    word_bank = unpacked_data(dataset_map[opts.wordbank])
                    updated_data = using_word_rep(data_to_rep, global_gen_number,  
                                                  word_to_id, word_embeds, word_bank, opts.rep_with, opts.append_yago)
                    
                assert len(updated_data)==len(data_to_rep), 'error'
                
                if opts.filter:
                    updated_data = filter_examples(dataset_map[opts.dataset],updated_data, number_to_generate)
                
                savefile(updated_data, opts, agg_name)

            print('rep finished')
                      
        if not opts.cascade:
            global_gen_number = number_to_generate
        


if __name__=="__main__":
    
    perplexity_filter = Perplexity_estimator()    
    main()


