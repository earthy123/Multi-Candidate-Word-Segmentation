# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline


import torch
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import math
import re
import pickle
import multiprocessing
import itertools
from sklearn.model_selection import KFold

n_fac =512
bs= 128
hidden_char= 128
hidden_layer = 512
n_classes = 2
lr =1e-4
c=23


def acc(outputs,labels,bs):
    total_acc=0.0
    count =0
    _, preds = torch.max(outputs.data, 1)
#     print('preds',preds)
#     print('labels',labels)
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
        loc =np.where(y == 1)[0]
        no_seg = len(loc)
        if no_seg >=1:

            for i in range(len(loc)):
                end=loc[i]+1
                check = np.any(np.logical_xor(preds[start:end], y[start:end]))
                start=loc[i]
            #     print(i)
            #     print(check)
                if check == False:
                    correct +=1
            acc = correct/len(loc)
        else:
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

def get_cv_idxs(n, cv_idx=0, val_pct=0.1, seed=42):
    np.random.seed(seed)
    n_val = int(val_pct*n)
    idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start+n_val]

def split_by_idx(idxs, *a):
    mask = np.zeros(len(a[0]),dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask],o[~mask]) for o in a]

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

with open('all_tcc.pkl', 'rb') as handle:

    uniques_tcc = pickle.load(handle)
    
with open('allcharEN-TH-SP.pkl', 'rb') as handle:

    allchar = pickle.load(handle)
    
with open('allchars.pkl', 'rb') as handle:

    chars = pickle.load(handle)

# PATH='allbest/'
# text = open(f'{PATH}allarticle_new.txt').read()
# import file text
text = open('mix1.txt').read()

# PATH='article/'
# text = open(f'{PATH}article_00001.txt').read()
print(len(text))
# chars = sorted(list(set(text)))

# text_tcc =tcc(text)
#text2tcc
text_tcc = parallel_tcc(text,10)
text_tcc = text_tcc[:-1]
text_tcc = text_tcc.split('*//*')
text_tcc = list(filter(None, text_tcc))
uniques_tcc.extend(chars)
tcc_indices = dict((c,i) for i,c in enumerate(uniques_tcc))
indices_tcc = dict((i,c) for i,c in enumerate(uniques_tcc))

char_indices = dict((c,i) for i,c in enumerate(chars))
indices_cahr = dict((i,c) for i,c in enumerate(chars))


#create x and y set function create_dataset is create for specific InterBEST2009/2010 NECTEC
xs,ys = create_dataset(text_tcc,uniques_tcc)

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
#apply loss functin
#nll_lose with log_softmax is equavalent to coss_entropy with softmax
def nll_loss_seq(inp,targ):
    bs,sl,nh = inp.size()

#     targ = targ.transpose(0,1).contiguous().view(-1)
    return F.nll_loss(inp.view(-1,nh),targ.view(-1))
#use Adam optimizer
optimizer =optim.Adam(model.parameters(), lr=lr)

#use an identity maxtrix to avoid gradient vanishing
model.lstm.weight_hh_l0.data.copy_(torch.eye(hidden_layer*4,hidden_layer))
model.lstm.weight_hh_l0_reverse.data.copy_(torch.eye(hidden_layer*4,hidden_layer))
#apply 5-kold cross-validation
kf = KFold(n_splits=5)
kf.get_n_splits(xs)
KFold(n_splits=5, random_state=None, shuffle=False)
#sliding windows by 1 with window size 23
c_in_datas = [[xs[i+j] for i in range(c)] for j in range(len(xs)-c+1)]
c_out_datas = [[ys[i+j] for i in range(c)] for j in range(len(ys)-c+1)]

x =np.stack(c_in_datas)
y = np.stack(c_out_datas)
xx=x
yy =y
val_len = x.shape[0]
val_idx = get_cv_idxs(val_len-c)

all_losses = []
iter = 0
plot_every = 15
num_epochs=4
k=1
# total_loss = 0
start = time.time()
for train_index,test_index in kf.split(xx):
    trn_xs, val_xs = xx[train_index], xx[test_index]
    trn_y, val_y = yy[train_index], yy[test_index]
    char_pad_trn_xs=gen_char_list(trn_xs)
    char_trn =np.stack(char_pad_trn_xs)
    char_pad_val_xs=gen_char_list(val_xs)
    char_val =np.stack(char_pad_val_xs)
    
    class trn_dataset(Dataset):
        def __init__(self):
            self.len = trn_xs.shape[0]
            self.x_data = torch.from_numpy(trn_xs)
            self.y_data = torch.from_numpy(trn_y)
            self.chars_data = torch.from_numpy(char_trn)

        def __getitem__(self,index):
            return self.x_data[index],self.y_data[index],self.chars_data[index]
        def __len__(self):
            return self.len

    class val_dataset(Dataset):
        def __init__(self):
            self.len = val_xs.shape[0]
            self.x_data = torch.from_numpy(val_xs)
            self.y_data = torch.from_numpy(val_y)
            self.chars_data = torch.from_numpy(char_val)

        def __getitem__(self,index):
            return self.x_data[index],self.y_data[index],self.chars_data[index]
        def __len__(self):
            return self.len

    train_loader =  DataLoader(dataset=trn_dataset(),batch_size=bs,shuffle=False)
    val_loader =  DataLoader(dataset=val_dataset(),batch_size=bs,shuffle=False)
    
    total_loss = 0
#     all_losses = []
    print('k-fold :',k)
    k +=1
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                loader =train_loader
                dataset_size= loader.dataset.len
            else:
                loader =val_loader
                dataset_size= loader.dataset.len
            running_loss = 0.0
            running_corrects = 0

            for data in loader:
                inputs,labels,chars =data
                inputs,labels,chars  = Variable(inputs).cuda(), Variable(labels).cuda(),Variable(chars).cuda()




                optimizer.zero_grad()

                outputs = model(inputs,chars,8)
    #             print(outputs)

    #             b_s =outputs.size()[0]
    #             print(b_s)



                loss = nll_loss_seq(outputs,labels)

    #             loss = criterion(outputs,labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                iter +=1                
                running_loss += loss.data[0]*inputs.size(0)
                total_loss += loss.data[0]*inputs.size(0)
                b_s =outputs.size()[0]


                running_corrects += acc(outputs,labels.data,b_s)
    #             print('run corre: ',running_corrects)
                if iter % plot_every == 0:
                    all_losses.append(total_loss / plot_every)
                    total_loss = 0

    #         print('runn cor: ',running_corrects)
            epoch_loss =( running_loss / dataset_size)
            epoch_acc = (running_corrects / dataset_size)
            print('{} Loss: {:.4f} Acc: {:.4f} Time: {}'.format(phase, epoch_loss,epoch_acc,(timeSince(start))))
            print()
    
    print()
#save model
torch.save(model.state_dict(), 'p-test23ws.pth')
torch.save(model, "test23ws.pth") 
with open('test23ws.txt', 'w') as file_handler:
    for item in all_losses:
        file_handler.write("{}\n".format(item))
print('DONE')



