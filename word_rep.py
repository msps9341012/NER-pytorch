from __future__ import print_function

import numpy as np

import itertools
import loader
from utils import *
from loader import *
from conlleval import split_tag, is_chunk_end, is_chunk_start
from scipy import spatial
import pickle

pool_method_ids = {"mean":0, "min":1, "max":2}
tags_all = ['LOC', 'MISC', 'PER', 'ORG']


class Word_Replacement():

    def __init__(self, lower, word_to_id, word_embeds):

        self.lower = lower
        self.map_tag_to_chunks = {}
        self.word_to_id = word_to_id
        self.word_embeds = word_embeds
        tags_all = ['LOC', 'MISC', 'PER', 'ORG']
        for tag in tags_all:
            self.map_tag_to_chunks[tag] = []

    def calculate_net_embedding_vector_for_chunk(self, chunk_to_replace, pool_method = "mean"):
        """
        """

        def f(x): return x.lower() if self.lower else x
    
        str_words = [ent[0] for ent in chunk_to_replace]
        word_ids = [self.word_to_id[f(w) if f(w) in self.word_to_id else '<UNK>']
                     for w in str_words]
    
        all_distances = [self.word_embeds[word_id] for word_id in word_ids]
    
        if pool_method == "min":
            embedding = np.min(np.array(all_distances),axis=0)
        elif pool_method == "max":
            embedding = np.max(np.array(all_distances),axis=0)
        else:
            embedding = np.mean(np.array(all_distances), axis=0)
    
        return embedding

    def create_tag_chunks(self, sentence):

        # """ 
        # Append the chunks in the 
        # """
        prev_tag = 'O'
        start_tag = False
        end_tag = False
        current_chunk = []
    
        for ent in sentence:
            tag = ent[3]
            end_tag = is_chunk_end(prev_tag, tag)
            if end_tag and len(current_chunk) > 0:
                _ , tag_type = split_tag(current_chunk[-1][3])
                chunk_embedding_min = self.calculate_net_embedding_vector_for_chunk(current_chunk, "min")
                chunk_embedding_max = self.calculate_net_embedding_vector_for_chunk(current_chunk, "max")
                chunk_embedding_mean = self.calculate_net_embedding_vector_for_chunk(current_chunk, "mean")
    
                self.map_tag_to_chunks[tag_type].append((current_chunk, chunk_embedding_mean, chunk_embedding_min, chunk_embedding_max))
                current_chunk = []
                start_tag = False
    
            if not start_tag:
                start_tag = is_chunk_start(prev_tag, tag)
            if start_tag:
                current_chunk.append(ent)
            prev_tag = tag

    def find_top_replacement(self, chunk_to_replace, max_replacements=10, pool_method = "mean", replacement_method = "closest"):

        top_replacements = []
    
        _, tag_type = split_tag(chunk_to_replace[0][3])
    
        base_embedding = self.calculate_net_embedding_vector_for_chunk(chunk_to_replace,  pool_method)
    
        possible_replacements = self.map_tag_to_chunks[tag_type] 
    
        pool_method_id = pool_method_ids[pool_method]
    
        if replacement_method == "random":
    
            for res_idx in range(max_replacements):
                top_replacements.append(possible_replacements[random.randint(0, len(possible_replacements)-1)][0])
    
        else:
            all_embeddings = [pr[pool_method_id+1] for pr in possible_replacements]
    
            # all_distances = [np.linalg.norm(base_embedding - embedding) for embedding in all_embeddings]
    
            all_distances = [spatial.distance.cosine(base_embedding, embedding) for embedding in all_embeddings]
    
            all_distances_np = np.array(all_distances) 
    
            if replacement_method == "farthest":
                sort_index = np.argsort(-all_distances_np)
            else:
                sort_index = np.argsort(all_distances)
    
            res_idx = 0
    
            while max_replacements > 0:
                if all_distances_np[sort_index[res_idx]] > 0:
                    top_replacements.append(possible_replacements[sort_index[res_idx]][0])
                    max_replacements -= 1
                res_idx = res_idx + 1
    
        return top_replacements

    def generate_adversarial_examples(self, word_seq, max_examples=10, pool_method = "mean", replacement_method = "closest"):
        """
    
    
        """
        # Storing map of seq_id to list of possible replacements
        replacement_dict = {}
        to_replace = {}
    
        adversarial_examples = []
    
        for seq_id in range(len(word_seq)):
            type_word = word_seq[seq_id][0]
            if type_word == 1:
                to_replace[seq_id] = word_seq[seq_id][1]
                replacement_dict[seq_id] = []
    
        if len(replacement_dict) == 0:
            max_examples = 1
    
        for key in replacement_dict:
            chunk_to_replace = to_replace[key]
            potential_replacements = self.find_top_replacement(chunk_to_replace, max_examples, pool_method, replacement_method)
            replacement_dict[key] = potential_replacements
    
        for example_idx in range(max_examples):
            adv_senetence = []
            for seq_id in range(len(word_seq)):
                type_word = word_seq[seq_id][0]
                if type_word == 0:
                    adv_senetence.append(word_seq[seq_id][1][0])
                else:
                    for item in replacement_dict[seq_id][example_idx]:
                        adv_senetence.append(item)
    
            adversarial_examples.append(adv_senetence)
    
        return adversarial_examples
    
    def create_adv_examples(self, sentence, max_examples=10, pool_method = "mean", replacement_method = "closest"):
        # """ 
        # Append the chunks in the 
        # """
        prev_tag = 'O'
        start_tag = False
        end_tag = False
        current_chunk = []
        
        # 0 : word from original sentence
        # 1 : word from adversarial replacement
        word_seq = []
    
        for idx in range(len(sentence)):
            ent = sentence[idx]
            tag = ent[3]
            end_tag = is_chunk_end(prev_tag, tag)
    
            if end_tag and len(current_chunk) > 0:
                _, tag_type = split_tag(current_chunk[-1][3])
                word_seq.append((1, current_chunk))
                current_chunk = []
                start_tag = False
    
            if not start_tag:
                start_tag = is_chunk_start(prev_tag, tag)
    
            if start_tag:
                current_chunk.append(ent)
            else:
                word_seq.append((0, [ent]))
            prev_tag = tag

        #print(word_seq)
        adversarial_examples = self.generate_adversarial_examples(word_seq, max_examples, pool_method, replacement_method)
        
        return adversarial_examples

def main():

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


    pre_emb = "/mnt/nfs/scratch1/nkhade/glove.6B.100d.txt"
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

    wr = Word_Replacement(lower, word_to_id, word_embeds)
    
    for sentence in train_sentences:
        wr.create_tag_chunks(sentence)

    print("Generating Closest Adversarial Examples")
    all_adversarial_examples_farthest = []

    print("Train Len : {} , Dev Len : {}".format(len(train_sentences), len(dev_sentences)))
    counter = 1

    for sentence in train_sentences:
        print(counter)
        adversarial_examples = wr.create_adv_examples(sentence, 1, "mean", "farthest")

        all_adversarial_examples_farthest = all_adversarial_examples_farthest + adversarial_examples
        counter += 1

    all_adversarial_farthest_data = prepare_dataset( all_adversarial_examples_farthest, word_to_id, char_to_id, tag_to_id, True )

    # all_adversarial_farthest_data_dict = dict(zip(all_adversarial_farthest_data,range(len(all_adversarial_farthest_data))))

    with open('rep_text/rep_token_agg_complete', 'wb') as handle:
        pickle.dump(all_adversarial_farthest_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('rep_text/rep_token_agg', 'rb') as handle:
    #     rep_token_list = pickle.load(handle)

if __name__=="__main__":
    main()