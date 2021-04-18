import pickle, pdb, random
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from tqdm import tqdm

def add_to_dict_of_set(key, value, dict_set):
    if key in dict_set:
        dict_set[key].add(value)
    else:
        dict_set[key] = {value}

def clean_paraphrase(paraphrase_dict):
    stemmer = SnowballStemmer("english")
    paraphrase_dict_clean = dict()
    print("Size: %d" % len(paraphrase_dict))

    for phrase, paraphrases in paraphrase_dict.items():
        new_paraphrases = set()
        for paraphrase in paraphrases:
            if stemmer.stem(phrase) != stemmer.stem(paraphrase):
                new_paraphrases.add(paraphrase)
        if len(new_paraphrases):
            paraphrase_dict_clean[phrase] = new_paraphrases
    print("Size: %d" % len(paraphrase_dict_clean))
    return paraphrase_dict_clean
        


class PPDB_Replacement():
    
    def __init__(self, filename, word_maps):
        self.paraphrase_dict=self.collect_pairs_by_rel(filename,word_maps)
        self.paraphrase_dict_clean = clean_paraphrase(self.paraphrase_dict)
        self.id_of_change_examples = set()
           
    def collect_pairs_by_rel(self, filename, word_maps):
        """ Collect pairs from PPDB maintaining the specified relation. """
        stemmer = SnowballStemmer("english")

        with open(filename, "r") as f:
            data = f.readlines()

        phrase2paraphrase = dict()

        for item in tqdm(data):
            item = item.strip()
            phrase = item.split('|||')[1].strip()
            #check phrase
            flag=0
            for word in phrase.split():
                if word.lower() not in word_maps:
                    flag=1
                    break
            if flag:
                continue

            #check paraphrase
            paraphrase = item.split('|||')[2].strip()
            flag=0
            for word in paraphrase.split():
                if word.lower() not in word_maps:
                    flag=1
                    break
            if flag:
                continue

            if stemmer.stem(phrase) == stemmer.stem(paraphrase):
                continue
            entailment = item.split('|||')[-1].strip()

            if entailment == 'Equivalence':
                add_to_dict_of_set(phrase, paraphrase, phrase2paraphrase)
                add_to_dict_of_set(paraphrase, phrase, phrase2paraphrase)

        print("Size: %d" % len(phrase2paraphrase))
        return phrase2paraphrase
    
    def gen_paraphrase_for_text(self, text, paraphrase_dict):
        """ This function replace several words/phrases in a sentence at once
        for generating paraphrases. """
        paraphrases = set()
        tokens = word_tokenize(text)
        replaced = []
        replacement = []
        token_idx = 0
        while token_idx < len(tokens):
            unigram = tokens[token_idx]
            if token_idx < len(tokens) - 1:
                bigram = tokens[token_idx] + " " + tokens[token_idx]
            else:
                bigram = None
            if bigram and bigram in paraphrase_dict:
                replaced.append(bigram)
                replacement.append(paraphrase_dict[bigram])
                token_idx += 1
            elif unigram in paraphrase_dict:
                replaced.append(unigram)
                replacement.append(paraphrase_dict[unigram])

            token_idx += 1
        # generate token possibilities
        num_paraphrases = min([len(replaced)] + [len(item) for item in replacement])
        for item_idx in range(len(replacement)):
            token_para = random.sample(replacement[item_idx], num_paraphrases)
            # if len(token_para) < num_paraphrases:
            #     token_para += [random.choice(token_para) for _ in range(num_paraphrases - len(token_para))]
            replacement[item_idx] = token_para

        # generate paraphrases
        for paraphrase_idx in range(num_paraphrases):
            new_text = text
            for token_replaced, token_para in zip(replaced, replacement):
                new_text = new_text.replace(token_replaced, token_para[paraphrase_idx])
            paraphrases.add(new_text)


        return paraphrases
    
    def get_para(self, words, index):
        res=[]
        para_sent=list(self.gen_paraphrase_for_text(" ".join(words), self.paraphrase_dict_clean))
        if para_sent:
            self.id_of_change_examples.add(index)
            words=para_sent[0].split()
        for i in words:
            res.append([i,'_','_','O']) 
        return res
    
    def para_examples(self, data, index):
        res=[]
        tmp=[]
        for word_list in data:
            if word_list[-1]=='O':
                tmp.append(word_list[0].lower())
            else:
                if tmp:
                    res.extend(self.get_para(tmp,index))
                    res.append(word_list)
                    tmp=[]
                else:
                    res.append(word_list)
        if tmp:
            res.extend(self.get_para(tmp,index))  
        return res
        


# paraphrase_dict=collect_pairs_by_rel('../ppdb/ppdb_all', pretrained_word_list)
# paraphrase_dict_clean = clean_paraphrase(paraphrase_dict)

# res=[]
# index=0
# change=set()
# for i in train_sentences:
#     res.append(para_example(i,index))
#     index=index+1
# print('Modify examples:' + str(len(change)/len(res)))


# with open('../para_text/para_ppdb_train', 'wb') as handle:
#     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


