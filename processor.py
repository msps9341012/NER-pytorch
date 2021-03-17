from collections import defaultdict
import numpy as np

def padding_att(att_list,length):
    tmp=[]
    for i in att_list:
        tmp.append(i+[0]*(max(length)-len(i)))
    return np.array(tmp)

def padding_char(att_list, length):
    tmp=[]
    for i in att_list:
        for j in i:
            tmp.append(j+[0]*(max(length)-len(j)))
    return np.array(tmp)


def get_batch(batch_data):
    batch=defaultdict(list)
    for data in batch_data:
        batch['chars'].append(data['chars'])
        batch['words'].append(data['words'])
        batch['tags'].append(data['tags'])
        batch['caps'].append(data['caps'])
        batch['char_length'].extend(list(map(lambda x: len(x), data['chars'])))
        batch['word_length'].append(len(data['words']))
        
    batch['words']=padding_att(batch['words'], batch['word_length'])
    batch['tags']=padding_att(batch['tags'], batch['word_length'])
    batch['caps']=padding_att(batch['caps'], batch['word_length'])
    batch['chars']=padding_char(batch['chars'], batch['char_length'])
    return batch

def generate_batch_data(data_list,batch_size):
    data_batch=[]
    for index in range(0,len(data_list),batch_size):
        data_batch.append(get_batch(data_list[index:index+batch_size]))
    return data_batch


def generate_batch_para(para_data, batch_size):
    
    batch_list=[]
    tmp_batch=[]
    for i in para_data:
        packed_data=get_batch(para_data[i])
        tmp_batch.append(packed_data)
        if len(tmp_batch)==batch_size:
            batch_list.append(tmp_batch)
            tmp_batch=[]
    if tmp_batch:
        batch_list.append(tmp_batch)
    return batch_list
            

def generate_batch_rep(rep_data, batch_size):
    
    batch_list=[]
    tmp_batch=[]
    for data in rep_data:
        if not data['str_words']:
            continue
        tmp_batch.append(data)
        if len(tmp_batch)==batch_size:
            packed_data=get_batch(tmp_batch)
            batch_list.append(packed_data)
            tmp_batch=[]
    if tmp_batch:
        packed_data=get_batch(tmp_batch)
        batch_list.append(packed_data)
    return batch_list
            
        
    
    
    
    
    
