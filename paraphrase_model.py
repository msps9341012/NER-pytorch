
import torch
import sys
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import dependency_paraphraser
import random
from loader import cap_feature

import dependency_paraphraser
#path = 'english-ewt-ud-2.5-191206.udpipe'

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
        for word in word_List:
            true_order.append(word[0])
            if word[-1].startswith('I') or word[-1].startswith('E'): #combine tags
                res[-1] = res[-1] + " " + word[0]
                tag_list[-1] = tag_list[-1] +" "+ word[-1]
            else:
                res.append(word[0])
                tag_list.append(word[-1])

        return res, tag_list, true_order


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
        text, tags, true_order=self.combine_tag(word_list[:-1])
        if len(text) == 1:
            return
        res = []
        for i in range(n):
            para_w, para_t = self.get_paraphrase(text,tags)
            
            assert len(para_w) == len(para_t), 'error'
            
            para_w += [word_list[-1][0]]
            para_t += [word_list[-1][-1]]
            
            if self.perplexity_score(" ".join(para_w))<285 and para_w != true_order:
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


    
    
    