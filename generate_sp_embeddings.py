from collections import OrderedDict
import numpy as np
import os
import glob
import pickle
from model_bl import D_VECTOR
import torch

# Line 10 to 16 : Code copied from https://github.com/auspicious3000/autovc
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)

speakers_data=list()
min_length=128

for file in glob.glob('./spmel/*'):

    speaker_data=list()
    speaker=file.split('./')[-1]
    speaker_data.append(speaker)
    speaker_embedding=list()
    for spec_file in glob.glob(file+'/*'):
        spec=np.load(spec_file)
        if spec.shape[0]<=min_length:
            continue
       
        max_possible_value=max(0,spec.shape[0]-min_length)
        start = np.random.randint(0,min(1500,max_possible_value))
        #print(spec_file,end=' ')
        #print(spec.shape)
        
        
        spec=torch.from_numpy(spec[np.newaxis, start:start+min_length, :]).cuda()
        tmp=C(spec).detach().squeeze().cpu().numpy()
        speaker_embedding.append(tmp)
    speaker_data.append(np.mean(speaker_embedding, axis=0))
    for spec_file in glob.glob(file+'/*'):
        speaker_data.append('/'.join(spec_file.split('/')[-2:]))
    speakers_data.append(speaker_data)

with open('./spmel/train.pkl','wb') as f:
    pickle.dump(speakers_data,f)

