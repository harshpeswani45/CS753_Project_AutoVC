
import os
import argparse
import model
import make_dataloader
import torch
import torch.nn.functional as F


device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
train_dl=make_dataloader.ret_loader()
G = model.EncoderDecoder(16,256,512)        
optimizer = torch.optim.Adam(G.parameters(), 0.0001)
G.to(device)

epochs=1000000

for i in range(epochs):
    for utt,sp in train_dl:
        utt=utt.to(device)
        sp=sp.to(device)
        out=G(utt,sp,sp)
        loss=F.mse_loss(utt,out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i%5==0:
        print(str(i)+'th epoch loss='+str(loss))
        torch.save({
            'model': G.state_dict()
            }, './model2.ckpt')
