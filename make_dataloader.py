from random import shuffle
from torch.utils.data import Dataset,DataLoader
from collections import OrderedDict
import numpy as np
import os
import glob
import pickle
from model_bl import D_VECTOR
import torch



#TODO: make uttr and crop uttr and speaker_emb
# Way is different a little bit

class AutoVC(Dataset):
    def __init__(self,uttr,speaker_emb):
       self.uttr=uttr 
       self.speaker_emb=speaker_emb

    def __len__(self):
        return len(self.uttr)
    
    def __getitem__(self, idx):
        #see to(device) or not
        return torch.tensor(self.uttr[idx]), torch.tensor(self.speaker_emb[idx])
    

def ret_loader():
    train_pkl=pickle.load(open('./spmel/train.pkl','rb'))
    uttrances=list()
    speaker_emb=list()
    for i in range(len(train_pkl)):
        for j in range(2,len(train_pkl[i])):
            #print('./spmel/'+train_pkl[i][j])
            tmp_uttr=np.load('./spmel/'+train_pkl[i][j])
            if tmp_uttr.shape[0] < 128:
                len_pad = 128 - tmp_uttr.shape[0]
                uttr = np.pad(tmp_uttr, ((0,len_pad),(0,0)), 'constant')
            elif tmp_uttr.shape[0] > 128:
                left = np.random.randint(tmp_uttr.shape[0]-128)
                uttr = tmp_uttr[left:left+128, :]
            else:
                uttr = tmp_uttr
            #print(uttr.shape)
            uttrances.append(uttr)
            speaker_emb.append(train_pkl[i][1])

    train_ds = AutoVC(uttrances,speaker_emb)
    train_dl = DataLoader(train_ds, batch_size=40,shuffle=True,drop_last=True)
    return train_dl

