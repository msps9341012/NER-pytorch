
import torch
import sys
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import dependency_paraphraser
import random
from loader import cap_feature
import string

import dependency_paraphraser
import re
from collections import Counter
#path = 'english-ewt-ud-2.5-191206.udpipe'


def check_single_quote(word):
    if re.match('\'\w+',word):
        return True
    return False

class Paraphraser():
    
    
    def __init__(self,path):
        with torch.no_grad():
            self.scoring_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.scoring_model.eval()
            self.scoring_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
        self.pipe = dependency_paraphraser.udpipe.Model(path)
        self.projector = dependency_paraphraser.udpipe.en_udpipe_projector
    
    def perplexity_score(self,sentence):
        '''
        sentence: sentence in string
        '''
        tokenize_input = self.scoring_tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input])
        loss = self.scoring_model(tensor_input, labels = tensor_input)[0]
        return np.exp(loss.detach().numpy())
     
    def combine_tag(self, word_List):
        '''
        word_list: a input example in the original format, [[word, pos_tag, chunck_tag, ner_tag]..]
        '''
        res = []
        tag_list = []
        true_order = []
        punct_counter=Counter()
         
        for word in word_List:
            true_order.append(word[0])
            if word[0] in string.punctuation and word[-1]=='O':
                punct_counter[word[0]]+=1
            if word[-1].startswith('I') or word[-1].startswith('E') or check_single_quote(word[0]): #combine tags
                res[-1] = res[-1] + " " + word[0]
                tag_list[-1] = tag_list[-1] +" "+ word[-1]
            else:
                res.append(word[0])
                tag_list.append(word[-1])

        return res, tag_list, true_order, punct_counter
    
    def get_paraphrase(self, text, tags, tree_temperature=1):
        '''
        text/tags: in list format
        '''
        text = '\n'.join(text)+'\n'
        words_para, tag_para = dependency_paraphraser.udpipe.paraphrase(text, self.pipe, projector = self.projector,
                                                                        tree_temperature = tree_temperature, ner_list = tags)
        return words_para, tag_para
    
    def generate_n_paraphrases(self, word_list, n):
        if len(word_list) < 6:
            return
        text, tags, true_order, punct_counter=self.combine_tag(word_list)
        if punct_counter['('] and punct_counter[')']:
            text, tags=self.handle_pair_punct(text,tags,'(')
        if punct_counter['['] and punct_counter[']']:
            text, tags=self.handle_pair_punct(text,tags,'[')

        if len(text)==1:
            return
        res = []
        true_order=' '.join(true_order)
        
        for i in range(n):
            para_w, para_t = self.get_paraphrase(text,tags)
            
            assert len(para_w) == len(para_t), 'error'
            para_text=" ".join(para_w)
            #para_w += [word_list[-1][0]]
            #para_t += [word_list[-1][-1]]
            
            if para_text!=true_order and self.perplexity_score(para_text)<285:
                res.append([para_w, para_t])
                
        return res
    
    def prepare_para_dataset(self, sentences, word_to_id, char_to_id, tag_to_id, lower=True):
        """
        sentences: list of [word_list, ner_list]
        Prepare the dataset. Return a list of lists of dictionaries containing:
            - word indexes
            - word char indexes
            - tag indexes
        """
        def f(x): return x.lower() if lower else x
        data = []
        for word_list, ner_list in sentences:
            str_words = word_list
            words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                     for w in str_words]
            # Skip characters that are not in the training set
            chars = [[char_to_id[c] for c in w if c in char_to_id]
                     for w in str_words]
            caps = [cap_feature(w) for w in str_words]
            tags = [tag_to_id[w] for w in ner_list]
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'caps': caps,
                'tags': tags,
            })
        return data
    
    
    def convert_span(self,text_list,punct):
        
        punct_map={'(':')',"[":']'}
        
        left=[]
        res=[]
        for i in range(len(text_list)):
            if text_list[i]==punct:
                left.append(i)
            elif text_list[i]==punct_map[punct]:
                if left:
                    res.append([left.pop(),i])
        if not res:
            return []
        if len(res)==1:
            return res
        res.sort()
        span=[res[0]]
        for l,r in res[1:]:
            if l>span[-1][-1]:
                span.append([l,r])
            else:
                span[-1][-1]=max(r,span[-1][-1])
        return span
    
    def handle_pair_punct(self, text_list, tags_list,punct):
        span=self.convert_span(text_list,punct)
        if not span:
            text_list, tags_list
        span=span[::-1]
        texts=[]
        tags=[]
        i=0
        while i<len(text_list):
            if span:
                if i<span[-1][0]:
                    texts=texts+[text_list[i]]
                    tags=tags+[tags_list[i]]
                    i=i+1
                elif i==span[-1][0]:
                    l,r=span.pop()
                    texts=texts+[' '.join(text_list[l:r+1])]
                    tags=tags+[' '.join(tags_list[l:r+1])]
                    i=r+1
            else:
                texts=texts+[text_list[i]]
                tags=tags+[tags_list[i]]
                i=i+1
        return texts, tags

    
    
    