from __future__ import print_function

import numpy as np
from tqdm import tqdm
import itertools
import loader
from utils import *
from loader import *
from conlleval import split_tag, is_chunk_end, is_chunk_start
from scipy import spatial
import pickle
import copy
import faiss
from collections import defaultdict
import torch
from transformers import BertModel, BertTokenizer
import sys



pool_method_ids = {"mean":0, "min":1, "max":2}
tags_all = ['LOC', 'MISC', 'PER', 'ORG']

class Bert:
    def __init__(self):
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
    def get_embedding(self,input_text):
        input_text = input_text.lower()
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
            
        return input_ids.cpu().detach().numpy(), last_hidden_states[0].cpu().detach().numpy()

    
class Embedding_Processor:
    def __init__(self, pooling_method):
        self.bert_model = Bert()
        self.map_tag_to_embed = defaultdict(dict)
        self.tag_string_to_chuck=defaultdict(dict)
        self.pooling_method = pooling_method 
    
        
     
    def create_tag_chunks(self, sentence):

            # """ 
            # Append the chunks in the 
            # """
            '''
            Assign index for the first subword in each tag. 
            (The index is based on tokenization result.)
            Use the index to extract embedding.
            The reason using this index is to prevent having the same subword in a sentence.
            '''
            start=1
            for ent in sentence:
                ids = self.bert_model.tokenizer.encode(ent[0], add_special_tokens=False)
                ent.append(start)
                start=start+len(ids)
                
            append_flag=0
            if sentence[-1][-1]!='O':
                sentence=sentence+[['.','_','_','O']]
                append_flag=1

            prev_tag = 'O'
            start_tag = False
            end_tag = False
            current_chunk = []
            tag_word_map=defaultdict(list)
            for ent in sentence:
                tag = ent[3]
                end_tag = is_chunk_end(prev_tag, tag)
                if end_tag and len(current_chunk) > 0:
                    _ , tag_type = split_tag(current_chunk[-1][3])
                    tag_word_map[tag_type].append(current_chunk)
                    current_chunk = []
                    start_tag = False

                if not start_tag:
                    start_tag = is_chunk_start(prev_tag, tag)
                if start_tag:
                    current_chunk.append(ent)
                prev_tag = tag


            if append_flag:
                sentence.pop(-1)

            self.get_embedding(sentence, tag_word_map)


    def convert_to_string(self, ent):
        string_list = [i[0] for i in ent]
        return " ".join(string_list).lower()

        
        
    def get_embedding(self, sentence, tag_word_map):
        sentence_string = self.convert_to_string(sentence)
        token_ids, emb = self.bert_model.get_embedding(sentence_string)

        token_ids =np.squeeze(token_ids)
        for tag_type in tag_word_map:
            words = tag_word_map[tag_type]
            for ent in words:
                tag_string = self.convert_to_string(ent)
                first_sub_word_index = ent[0][-1]
                if len(ent)==1:
                    tags_emb = emb[first_sub_word_index]
                else:
                    if self.pooling_method=='max':
                        tags_emb = emb[first_sub_word_index:first_sub_word_index+len(ent)].max(axis=0)
                    elif self.pooling_method=='mean':
                         tags_emb = emb[first_sub_word_index:first_sub_word_index+len(ent)].mean(axis=0)
                    else:
                        tags_emb = emb[first_sub_word_index]
                
                if tag_string in self.map_tag_to_embed[tag_type]:
                    self.map_tag_to_embed[tag_type][tag_string].append(tags_emb)

                else:
                    self.map_tag_to_embed[tag_type][tag_string] = [tags_emb]
                    #remove tokenization index, make the format as ususal
                    for i in ent:
                        i.pop(-1)
                    self.tag_string_to_chuck[tag_type][tag_string]= ent
        
         

    def pooling(self):
        for tag_type in self.map_tag_to_embed:
            for word in self.map_tag_to_embed[tag_type]:
                embs = self.map_tag_to_embed[tag_type][word]
                embs = np.array(embs).mean(axis=0)
                self.map_tag_to_embed[tag_type][word] = embs

        return self.map_tag_to_embed
            
        
    
    
         
        

    
def main():
    '''
    The input format should follow the training/dev sentences.
    [[[word,_,_,tag],...]]
    
    The output will be two dicts stored in two files.
    1. tag_string (raw_text) -> embedding vector
    2. tag_string (raw_text) -> chunk ([[word,..,tag],...]) 
    
    '''
    
    dataset = sys.argv[1]
    pooling_method = sys.argv[2]
    
    ep = Embedding_Processor(pooling_method)
    
    lower = 1
    zeros = 0
    tag_scheme = 'iobes'
    word_dim = 100

    train_sentences = loader.load_sentences('dataset/eng.train', lower, zeros)
    dev_sentences = loader.load_sentences("dataset/eng.testa", lower, zeros)
    test_sentences = loader.load_sentences("dataset/eng.testb", lower, zeros)

    update_tag_scheme(train_sentences, tag_scheme)
    update_tag_scheme(dev_sentences, tag_scheme)
    update_tag_scheme(test_sentences, tag_scheme)
    
    dataset_map={'train':train_sentences, 'dev':dev_sentences, 'test':test_sentences}
    
    print('Get bert embedding for {} using {}'.format(dataset,pooling_method))
    for i in tqdm(dataset_map[dataset]):
        ep.create_tag_chunks(i)
    
    #average same tags
    map_tag_to_embed = ep.pooling()

    
    if not os.path.exists('../tag_embed'):
        os.makedirs('../tag_embed')
        
    
    
    with open('../tag_embed/{}_bert_{}'.format(dataset,pooling_method), 'wb') as handle:
        pickle.dump(map_tag_to_embed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('../tag_embed/{}_bert_{}_chunck_map'.format(dataset,pooling_method), 'wb') as handle:
        pickle.dump(ep.tag_string_to_chuck, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


    
    
if __name__=="__main__":
    main()
