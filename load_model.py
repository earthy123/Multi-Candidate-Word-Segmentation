import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.backends.cudnn.enabled=False
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import numpy as np
import time
import math
import re
import pickle
import multiprocessing
import itertools
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support



with open('all_tcc.pkl', 'rb') as handle:

    uniques_tcc = pickle.load(handle)
    
with open('allcharEN-TH-SP.pkl', 'rb') as handle:

    allchar = pickle.load(handle)
    
with open('allchars.pkl', 'rb') as handle:

    chars = pickle.load(handle)
    
    
uniques_tcc.extend(chars)
tcc_indices = dict((c,i) for i,c in enumerate(uniques_tcc))
indices_tcc = dict((i,c) for i,c in enumerate(uniques_tcc))

char_indices = dict((c,i) for i,c in enumerate(chars))
indices_cahr = dict((i,c) for i,c in enumerate(chars))

    
    
    
def acc(outputs,labels,bs):
    total_acc=0.0
    count =0
    _, preds = torch.max(outputs.data, 1)
#     print('preds',preds)
#     print('labels',labels)
"""
preds is in batch form, we need to split each of them to compare with output
"""
    for i in range(bs):
        _, preds = torch.max(outputs[i].data, 1)
        y=labels[i]
#         labels[0].size


        #need preds,labels.data
        preds=preds.cpu().numpy()
        y=y.cpu().numpy()

        ##numpy not tensor
        acc=0.0
        start=0
        correct=0.0
        # check number of label 1 and location of it
        loc =np.where(y == 1)[0]
        no_seg = len(loc)
        if no_seg >=1:

            for i in range(len(loc)):
                #compare preds values with y values if reutrn false mean it exact same y so +1 of correct
                #it is word recall measuremnet because we use y is divider
                end=loc[i]+1
                check = np.any(np.logical_xor(preds[start:end], y[start:end]))
                start=loc[i]
            #     print(i)
            #     print(check)
                if check == False:
                    correct +=1
            acc = correct/len(loc)
        else:
            #in case 1 or 0 boudnary
            check = np.any(np.logical_xor(preds, y))
            if check == True:

                acc=0
            else:
                acc=1
            
        total_acc += acc
    return total_acc

def split_text(n):
    return n.split('|')

def parallel_tcc(text,chunk_size):
    chunk =[]
    #we don't want to randomly split into chunk, we want to split after "|"
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

# create validate set
def get_cv_idxs(n, cv_idx=0, val_pct=0.1, seed=42):
    np.random.seed(seed)
    n_val = int(val_pct*n)
    idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start+n_val]

def split_by_idx(idxs, *a):
    """
    create 0 matrices size a
    when index array match mark true
    return 90 of mask of false and 10 mask of true
    """
    mask = np.zeros(len(a[0]),dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask],o[~mask]) for o in a]



def create_dataset(inputs,chars):
    indices_cahr = dict((i,c) for i,c in enumerate(chars))
    char_indices = dict((c,i) for i,c in enumerate(chars))
    #convert char to idices
    idx=[char_indices[c] for c in inputs]
    #convert array to numpy
    idx =np.stack(idx)
    len_idx = len(idx)
    #keep number of "|"
    cutSymbol = char_indices['|']
    #create mask equal size to char type bool
    mask = np.zeros(len_idx,dtype=bool)
    loc =np.where(idx == cutSymbol)[0]
    #In input we have "|" but we want y thail|land but we want thailand and | in i
    delLoc=loc-1
    #mark true when found 1 
    mask[loc] = True
    new_mask = np.delete(mask,delLoc)
    #convert True,False into 1,0
    new_mask = new_mask.astype(int)
    new_idx = np.delete(idx,loc)
    return new_idx,new_mask

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)

def generate_char(tcc2char,MAX_WORD_LEN=8):
    """
    we set max word 8 because we know that maximum is smaller than 8
    it will useful when we want to split matrix because we already know that is form is 8*11
    """
    #need text after tcc
    sent_chars = []  
    for w in tcc2char:  

        sps = ' ' * (MAX_WORD_LEN - len(w))  
        sent_chars.extend(list(sps + w) if len(w) < MAX_WORD_LEN else list(w[:MAX_WORD_LEN]))  
    return sent_chars

def gen_char_list(x):
    collect_chars=[]
    for data in x:
        each_tcc = [indices_tcc[i] for i in data]
        chars_pad = generate_char(each_tcc)
        idx_chars =[char_indices[i] for i in chars_pad]
        collect_chars.append(idx_chars)
    return collect_chars

def confident_ranking(inputs,percent):
    prob_list =[]
    inputs_sort = sorted(inputs,reverse=True)
    select_inputs = inputs_sort[:int(len(inputs_sort)*percent)]
    for i in inputs:
        if i in select_inputs:
            prob_list.append(1)
        else:
            prob_list.append(0)
    return prob_list


pat_list = """\
เc็c
เcctาะ
เccีtยะ
เccีtย(?=[เ-ไก-ฮ]|$)
เccอะ
เcc็c
เcิc์c
เcิtc
เcีtยะ?
เcืtอะ?
เc[ิีุู]tย(?=[เ-ไก-ฮ]|$)
เctา?ะ?
cัtวะ
c[ัื]tc[ุิะ]?
c[ิุู]์
c[ะ-ู]t
c็
ct[ะาำ]?
แc็c
แcc์
แctะ
แcc็c
แccc์
โctะ
[เ-ไ]ct
ๆ
ฯลฯ
ฯ
""".replace('c','[ก-ฮ]').replace('t', '[่-๋]?').split()
'''
def tcc(w):
    p = 0 # position
    while p<len(w):
        for pat in pat_list:
            m = re.match(pat, w[p:])
            if m:
                n = m.span()[1]
                break
            else: # กรณีหาไม่เจอ
                n = 1
        yield w[p:p+n]
        p += n
'''
def tcc1(w):
    p = 0
    pat = re.compile("|".join(pat_list))
    while p<len(w):
        m = pat.match(w[p:])
        if m:
            n = m.span()[1]
        else:
            n = 1
        yield w[p:p+n]
        p += n
def tcc(w, sep='*//*'):
    return sep.join(tcc1(w))

def find_number_segment(inputs):
    
    inputs = np.stack(inputs)
    #finding location of 1
    location = np.where(inputs == 1)[0]
    num_segment =len(location)-1
    return location,num_segment

def find_number_segment_b(inputs):
    #finding location of 1
    #we need total number of boundaries
    inputs = np.stack(inputs)
    location = np.where(inputs == 1)[0]
    num_segment =len(location)
    return location,num_segment

def find_number_segment_w(inputs):
    #word need to add from begining and last digit
    inputs = np.stack(inputs)
    #we replace first and last element to 1
    #we -1 because 3 words contains 2 boundaries
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
    #correct output from predict is numerator
    #recall the divider is from y
    l_y,n_y = find_number_segment_b(y)
    #boundary check '|' symbols so n_y need to +1 cuz one word have 2 boundary
#     n_y +=1
    l_yh,_ = find_number_segment(predict)
    correct_predict =set(l_y).intersection(set(l_yh))
    recall = len(correct_predict)/n_y
    return recall

def boundary_precision(y,predict):
    #correct output from predict is numerator
    #recall the divider is from prediction
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
    #correct output from predict is numerator
    #recall the divider is from y
    l_y,n_y = find_number_segment_w(y)
    l_yh,n_yh = find_number_segment_w(predict)
    l_y_word = match_word(l_y)
    l_yh_word = match_word(l_yh)
    num_match_word =set(l_y_word).intersection(set(l_yh_word))
    result = len(num_match_word)/len(l_y_word)
    return result

def word_precision(y,predict):
    #correct output from predict is numerator
    #recall the divider is from prediction
    #to avoid error that divider is 0 we apply try and except
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




def predict(text,precentage_list):
    #convert to tcc
    txt_tcc = tcc(text)
    txt_tcc = txt_tcc.split('*//*')
    txt_tcc = list(filter(None, txt_tcc))
    #tcc 2 index
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
    return out
                   
                   
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
        len_inputs = len(inputs)
        sub =inputs[:,0] -inputs[:,1]
        inputs_sort =np.argsort(sub)
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
def one_candidate(text1):
    text_tcc = parallel_tcc(text1,10)
    text_tcc = text_tcc[:-1]
    text_tcc = text_tcc.split('*//*')
    text_tcc = list(filter(None, text_tcc))
    #we want ys
    xs,ys = create_dataset(text_tcc,uniques_tcc)
    idx=xs
    xs =[indices_tcc[i] for i in xs]
    list_txt =list(text1)
    new_tcc = list(filter(lambda x: x!= '|', list_txt))
    new_tcc = ''.join(new_tcc)
    #Seperate into chunks
    if len(xs) < 23:
        result = predict(text1,[0.9])
    else:
        
        win_size = 11
        c=win_size+1+win_size
        c_in_datas = [[xs[i+j] for i in range(c)] for j in range(len(xs)-c+1)]
        result = result_chunk(c_in_datas)
    result = np.stack(result)
    result=np.argmax(result, axis=1)
    i = np.stack(result)
    loc =np.where(i == 1)[0]
    loc +=1
    original_text = idx
    cut_symbol = tcc_indices['|']
    final_text =np.insert(original_text, loc, cut_symbol)
    cut_word = [(indices_tcc[i]) for i in final_text]
    cut_word = ''.join(cut_word)
    #         cut_word = cut_word.replace('\n','')
    print(cut_word)
    
def multi_candidate(text1):
    text_tcc = parallel_tcc(text1,10)
    text_tcc = text_tcc[:-1]
    text_tcc = text_tcc.split('*//*')
    text_tcc = list(filter(None, text_tcc))
    #we want ys
    xs,ys = create_dataset(text_tcc,uniques_tcc)
    idx=xs
    xs =[indices_tcc[i] for i in xs]
    list_txt =list(text1)
    new_tcc = list(filter(lambda x: x!= '|', list_txt))
    new_tcc = ''.join(new_tcc)
    #Seperate into chunks
    if len(xs) < 23:
        result = predict(text1,[0.9])
    else:
        
        win_size = 11
        c=win_size+1+win_size
        c_in_datas = [[xs[i+j] for i in range(c)] for j in range(len(xs)-c+1)]
        result = result_chunk(c_in_datas)
    result = np.stack(result)
    preds = con_ranking(result)  
    for i in preds:
        i = np.stack(i)
        loc =np.where(i == 1)[0]
        loc +=1
        original_text = idx
        cut_symbol = tcc_indices['|']
        final_text =np.insert(original_text, loc, cut_symbol)
        cut_word = [(indices_tcc[i]) for i in final_text]
        cut_word = ''.join(cut_word)
        #         cut_word = cut_word.replace('\n','')
        print(cut_word)
        
        


n_fac =512
bs= 128
hidden_char= 128
hidden_layer = 512
n_classes = 2
lr =1e-4
c=23



class lstm(nn.Module):
    def __init__(self,vocab_size,char_size,hidden_char,hidden_dim,n_classes):
        super(lstm,self).__init__()
        self.n_classes = n_classes
        self.hidden_char = hidden_char
        self.hidden_dim = hidden_dim
#         self.hidden = self.init_hidden(bs)
        
        self.word_embed = nn.Embedding(vocab_size,n_fac)
        self.char_embed = nn.Embedding(char_size,n_fac)
        self.char_lstm = nn.LSTM(n_fac, hidden_char,dropout=0.5,bidirectional=True)
        self.c_out = nn.Linear(hidden_char*2,n_fac)
        self.lstm = nn.LSTM(n_fac*2,hidden_dim,batch_first=True,dropout=0.5,bidirectional=True)
        self.out = nn.Linear(hidden_dim*2,n_classes)
        self.init_hidden(bs)
        self.initHidden_char(c*bs)
    
    def forward(self,cs,char_sent,MAX_WORD_LEN):
        bs = cs.size()[0]
        seq_len =cs[0].size(0)
        if self.h[0].size(0) !=bs :
            self.init_hidden(bs)
            self.initHidden_char(seq_len*bs)
            
            
        
#         print('bs',bs)
#         print('seq_len',seq_len)
        char_embed = self.char_embed(char_sent)
        word_embed = self.word_embed(cs)
        

        self.char_lstm.flatten_parameters()
        char_lstm_out, hc = self.char_lstm(char_embed.view(MAX_WORD_LEN,bs*seq_len,-1),self.h_c)
#         self.h_c= repackage_var(hc)
        c_out = F.relu(self.c_out(char_lstm_out))

        char_embeded=c_out[-1,:,:].view(-1,seq_len,n_fac)

        embeded = torch.cat((word_embed, char_embeded),dim=2) 

        self.lstm.flatten_parameters()
        out_lstm,h = self.lstm(embeded.view(-1,seq_len,n_fac*2),self.h)
#         self.h = repackage_var(h)
        out = self.out(out_lstm)
        
        return F.log_softmax(out,dim=-1)

    def initHidden_char(self,seq_len):
        
        self.h_c = (Variable(torch.zeros(2, seq_len, self.hidden_char)).cuda(),  
                  Variable(torch.zeros(2, seq_len, self.hidden_char)).cuda())  
        
    
    def init_hidden(self,bs):
        self.h =(autograd.Variable(torch.zeros(2,bs,self.hidden_dim)).cuda(),autograd.Variable(torch.zeros(2,bs,self.hidden_dim)).cuda())
        
        

model = lstm(len(uniques_tcc),len(chars),hidden_char,hidden_layer,n_classes).cuda()
model.cuda()
if __name__ == '__main__':
    model.load_state_dict(torch.load('p-test23ws.pth'))
    model = torch.load('test23ws.pth')
    print('load finnish')