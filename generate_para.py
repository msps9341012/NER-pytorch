import pickle
from paraphrase_model import Paraphraser

paraphraser=Paraphraser('english-ewt-ud-2.5-191206.udpipe')
lower = 1
zeros = 0
tag_scheme = 'iobes'

import loader

train_sentences = loader.load_sentences('dataset/eng.train', lower, zeros)
dev_sentences = loader.load_sentences("dataset/eng.testa", lower, zeros)
test_sentences = loader.load_sentences("dataset/eng.testb", lower, zeros)

from utils import *
from loader import *


update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

pre_emb="../glove_emb/glove.6B.100d.txt"
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



map_dict={}
index=0
for i in train_sentences:
    res=paraphraser.generate_n_paraphrases(i,5)
    if res:
        map_dict[index]=res
    index=index+1
    print(index)


token_map={}
for index, text_list in map_dict.items():
    token=paraphraser.prepare_para_dataset(text_list,word_to_id, char_to_id, tag_to_id)
    token_map[index]=token
    
with open('../para_text/para_token_agg', 'wb') as handle:
    pickle.dump(token_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
