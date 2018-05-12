import torch
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.backends.cudnn.enabled=False
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import math
import re
import pickle
import multiprocessing
import itertools
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
t0 = time.time()
def find_number_segment(inputs):
    
    inputs = np.stack(inputs)
    location = np.where(inputs == 1)[0]
    num_segment =len(location)-1
    return location,num_segment

def find_number_segment_b(inputs):
    
    inputs = np.stack(inputs)
    location = np.where(inputs == 1)[0]
    num_segment =len(location)
    return location,num_segment

def find_number_segment_w(inputs):
    #word need to add from begining and last digit
    inputs = np.stack(inputs)
    #we replace first and last element to 1
    np.put(inputs,[0,-1],[1,1])
    location = np.where(inputs == 1)[0]
    num_segment =len(location)-1
    return location,num_segment

def match_word(location):
    slot=[]
    location = list(location)
    for start,stop in zip(location,location[1:]+location[:1]):
        slot.append((start,stop))

    slot.pop(-1)
    return slot

def boundary_recall(y,predict):
    l_y,n_y = find_number_segment_b(y)
    #boundary check '|' symbols so n_y need to +1 cuz one word have 2 boundary
#     n_y +=1
    l_yh,_ = find_number_segment(predict)
    correct_predict =set(l_y).intersection(set(l_yh))
    recall = len(correct_predict)/n_y
    return recall

def boundary_precision(y,predict):
    check_all_zeros = np.stack(predict)
    zeros =np.all(check_all_zeros==0)
    if zeros == True:
        recall = 0
        return recall
    else:
        
        l_y,n_y = find_number_segment(y)
        l_yh,n_yh = find_number_segment_b(predict)
#         n_yh +=1
        correct_predict =set(l_y).intersection(set(l_yh))
        recall = len(correct_predict)/n_yh

        return recall

def word_recall(y,predict):
    l_y,n_y = find_number_segment_w(y)
    l_yh,n_yh = find_number_segment_w(predict)
    l_y_word = match_word(l_y)
    l_yh_word = match_word(l_yh)
    num_match_word =set(l_y_word).intersection(set(l_yh_word))
    result = len(num_match_word)/len(l_y_word)
    return result

def word_precision(y,predict):
    try:
        
        l_y,n_y = find_number_segment_w(y)
        l_yh,n_yh = find_number_segment_w(predict)
        l_y_word = match_word(l_y)
        l_yh_word = match_word(l_yh)
        num_match_word =set(l_y_word).intersection(set(l_yh_word))
        result = len(num_match_word)/len(l_yh_word)
        return result
    except:
        return 0
def f1(precision,recall):
    try:
        
        return 2*((precision*recall)/(precision+recall))
    except:
        return 0
    
def multi_candidate(result,percent_list=None):
    if percent_list is None:
        percent_list= [0.5]
    all_location=[]
    for j in percent_list:
        location=[]
        percent = j
        for i in range(len(result)):

            cut = (result[i][1])
            if cut>= percent:
                location.append(1)
            else:
                location.append(0)
        all_location.append(location)
    return all_location

def confident_ranking(inputs,percent):
    def yield_confident_ranking(inputs,percent):

        inputs =inputs[:,1]
        inputs_sort = sorted(inputs,reverse=True)
        select_inputs = inputs_sort[:int(len(inputs_sort)*percent)]
        for i in inputs:
            if i in select_inputs:
                yield 1
            else:
                yield 0
    return list(yield_confident_ranking(inputs,percent))

def multi_confident(result,percent_list):
    p= multiprocessing.Pool()
    percent_list = np.stack(percent_list)
    percent_list = 1-percent_list
    
    collection_confident =p.starmap(confident_ranking, zip(repeat(result),percent_list))
    p.close()
    p.join()
    return collection_confident
    
with open('all_tcc.pkl', 'rb') as handle:

    uniques_tcc = pickle.load(handle)
    
with open('allcharEN-TH-SP.pkl', 'rb') as handle:

    allchar = pickle.load(handle)
    
with open('allchars.pkl', 'rb') as handle:

    chars = pickle.load(handle)
    
uniques_tcc.extend(chars)

def create_dataset(inputs,chars):
    indices_cahr = dict((i,c) for i,c in enumerate(chars))
    char_indices = dict((c,i) for i,c in enumerate(chars))
    idx=[char_indices[c] for c in inputs]
    idx =np.stack(idx)
    len_idx = len(idx)
    cutSymbol = char_indices['|']
    mask = np.zeros(len_idx,dtype=bool)
    loc =np.where(idx == cutSymbol)[0]
    delLoc=loc-1
    mask[loc] = True
    new_mask = np.delete(mask,delLoc)
    new_mask = new_mask.astype(int)
    new_idx = np.delete(idx,loc)
    return new_idx,new_mask

def parallel_tcc(text,chunk_size):
    chunk =[]
    text =text.split('|')
    for i in range(0,len(text),chunk_size):
        c=text[i:i+chunk_size]
        c = '|'.join(c)+'|'
        chunk.append(c)
#     chunk = (lambda text,chunk_size: [text[i:i + chunk_size] for i in range(0,len(text), chunk_size)])(text,chunk_size)
    p= multiprocessing.Pool()
    result = p.map(tcc,chunk)
    p.close()
    p.join()
    result = '*//*'.join(result)
    return result

def predict(text,precentage_list):
    txt_tcc = tcc(text)
    txt_tcc = txt_tcc.split('*//*')
    txt_tcc = list(filter(None, txt_tcc))
    idx =[tcc_indices[i] for i in txt_tcc]
    text_var = autograd.Variable(torch.LongTensor(idx)).cuda()
    text_var = text_var.view(1,-1)
    sen_chars = generate_char(txt_tcc)
    char_var =[char_indices[i] for i in sen_chars]
    char_var = autograd.Variable(torch.LongTensor(char_var)).cuda()
    char_var = char_var.view(1,-1)
    outputs = model(text_var,char_var,8)
    outputs = outputs.squeeze()
    out = outputs.data.cpu().numpy()
#     print(out)
    div =out.sum(axis=1)
    result_div= out / div[:,None]
    result=1-result_div
    return result
                   
                   
def result_chunk(chunks):
    
    collection_result=[]
    len_c_in_datas = len(chunks)
    if len_c_in_datas <=11:
        for i,data in enumerate(chunks):
            flow =''.join(chunks[i])
            result = predict(flow,[0.9])
            if i ==0:
                collection_result.extend(result)
            elif (i == (len_c_in_datas-1 )):
                collection_result.extend(result[-(len_c_in_datas-1):])
            else:
                pass
    else:
        
        for i,data in enumerate(chunks):
            flow =''.join(chunks[i])
            result = predict(flow,[0.9])
            if i ==0:
                collection_result.extend(result)
            elif 1 <= i <= win_size:
                pass
            elif (len_c_in_datas-(win_size+1)) <= i <= (len_c_in_datas-2 ):
                pass
            elif (i == (len_c_in_datas-1 )):
                collection_result.extend(result)
            else:
                collection_result.append(result[win_size])
    return collection_result

def con_ranking(inputs):
    def yconfident_ranking(inputs):
        inputs =inputs[:,1]
        len_inputs = len(inputs)
        #max to low by index
        inputs_sort =np.argsort(inputs)[::-1]
        mask = np.zeros(len_inputs,dtype=bool)
        for i in range(1,len_inputs+1):
            mask[inputs_sort[:i]]=True
            new_mask = mask.astype(int)
            new_mask = list(new_mask)
            yield new_mask
    return list(yconfident_ranking(inputs))

def con_ranking1(inputs,chunk_size):
    merge_list=[]
    inputs =inputs[:,1]
    len_inputs = len(inputs)
    inputs_sort =np.argsort(inputs)[::-1]
    mask = np.zeros(len_inputs,dtype=bool)
    pointer= [i for i in range(1,len_inputs+1)]
    chunk = (lambda text,chunk_size: [text[i:i + chunk_size] for i in range(0,len(text), chunk_size)])(pointer,chunk_size)
    p= multiprocessing.Pool()
    result=p.starmap(seperate_work,zip(chunk,repeat(mask),repeat(inputs_sort)))
#     result = p.map(seperate_work,chunk)
    p.close()
    p.join()
    for i in result:
        merge_list.extend(i)
    return merge_list

def seperate_work(chunk,mask,inputs_sort):
    keep=[]
    for i in chunk:
        mask[inputs_sort[:i]]=True
        new_mask = mask.astype(int)
        new_mask = list(new_mask)
        keep.append(new_mask)
    return keep
                   
def multi_cand_word_recall(y,thresolds):
    
    l_y,n_y = find_number_segment_w(y)
    l_y_word = match_word(l_y)
    check_word=[]
    for i in thresolds:
        l_yh,n_yh = find_number_segment_w(i)
        l_yh_word = match_word(l_yh)
        num_match_word =set(l_y_word).intersection(set(l_yh_word))
        check_word.extend(list(num_match_word))

    match_result = set(l_y_word).intersection(set(check_word))
    result=len(match_result)/n_y
#     print(len(match_result))
#     print(n_y)
    return result

#mc word precision
def multi_cand_word_precision(y,thresolds):
    
    l_y,n_y = find_number_segment_w(y)
    l_y_word = match_word(l_y)
    check_word=[]
    total_match_word=[]
    for i in thresolds:
        l_yh,n_yh = find_number_segment_w(i)
        l_yh_word = match_word(l_yh)
        num_match_word =set(l_y_word).intersection(set(l_yh_word))
        check_word.extend(list(num_match_word))
        total_match_word.extend(l_yh_word)

    posible_words = set(total_match_word)
    match_result = set(l_y_word).intersection(set(check_word))
    result=len(match_result)/len(posible_words)
#     print(len(match_result))
#     print(len(posible_words))
    return result

#mc boundary recall
def multi_cand_boundary_recall(y,thresolds):
    last = thresolds[-1]
    l_last,n_last =find_number_segment_b(last)
    l_y,n_y = find_number_segment_b(y)
    total_seg = set(l_y).intersection(set(l_last))
    result =len(total_seg)/n_y
#     print(len(total_seg))
#     print(n_y)
    return result
#mc boudary precision
def multi_cand_boundary_precision(y,thresolds):
    last = thresolds[-1]
    l_last,n_last = find_number_segment_b(last)
    l_y,n_y = find_number_segment_b(y)
    total_seg = set(l_y).intersection(set(l_last))
    result =len(total_seg)/n_last
#     print(len(total_seg))
#     print(n_last)
    return result
def con_ranking(inputs):
    def yconfident_ranking(inputs):
        inputs =inputs[:,1]
        len_inputs = len(inputs)
        #max to low by index
        inputs_sort =np.argsort(inputs)[::-1]
        mask = np.zeros(len_inputs,dtype=bool)
        for i in range(1,len_inputs+1):
            mask[inputs_sort[:i]]=True
            new_mask = mask.astype(int)
            new_mask = list(new_mask)
            yield new_mask
    return list(yconfident_ranking(inputs))

#chnage to your dataset first
text = open('testset.txt').read()

print('text len: ',len(text))
text1 =text
text_tcc = parallel_tcc(text1,10)
text_tcc = text_tcc[:-1]
text_tcc = text_tcc.split('*//*')
text_tcc = list(filter(None, text_tcc))
print('seperate into tcc')
#we want ys
xs,ys = create_dataset(text_tcc,uniques_tcc)
xs =[indices_tcc[i] for i in xs]
list_txt =list(text1)
new_tcc = list(filter(lambda x: x!= '|', list_txt))
new_tcc = ''.join(new_tcc)
print('seperate into chunk')
##Seperate into chunks
win_size = 11
c=win_size+1+win_size
c_in_datas = [[xs[i+j] for i in range(c)] for j in range(len(xs)-c+1)]
result = result_chunk(c_in_datas)
with open('resultfromtest.pkl', 'wb') as handle:

    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('resultfromtest.pkl', 'rb') as handle:

    result = pickle.load(handle)

result = np.stack(result)
print('seperate into multi')
preds = con_ranking(result)              
y =list(ys)



br = multi_cand_boundary_recall(y,preds)
bp = multi_cand_boundary_precision(y,preds)
f1b= f1(bp,br)
wr =multi_cand_word_recall(y,preds)
wp = multi_cand_word_precision(y,preds)
f1w= f1(wp,wr)

t1 = time.time()
total = t1-t0
print(total)
