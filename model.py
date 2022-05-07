# Code copied from https://github.com/auspicious3000/autovc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self,dimension_bottle, embedding_dimension):
        super(Encoder, self).__init__()
        self.dimension_bottle = dimension_bottle
        self.embedding_dimension=embedding_dimension
        
        
        self.convolution_layer1 = torch.nn.Conv1d(80+embedding_dimension, 512,
                                    kernel_size=5, stride=1,
                         padding=2,
                         dilation=1,
                        bias=True)
        
        self.convolution_layer2 = torch.nn.Conv1d(512, 512,
                                    kernel_size=5, stride=1,
                         padding=2,
                         dilation=1,
                        bias=True)
        
        self.convolution_layer3 = torch.nn.Conv1d(512, 512,
                                    kernel_size=5, stride=1,
                         padding=2,
                         dilation=1,
                        bias=True)
            
        
        self.lstm = nn.LSTM(512, self.dimension_bottle, 2, batch_first=True, bidirectional=True)

    def forward(self, x, speaker):
        x = x.squeeze(1)
        x = x.transpose(2,1)
        speaker = speaker.unsqueeze(-1)
        speaker=speaker.expand(-1, -1, x.size(-1))
        x = torch.cat((x, speaker), dim=1)
        
        
        x = F.relu(self.convolution_layer1(x))
        x = F.relu(self.convolution_layer2(x))
        x = F.relu(self.convolution_layer3(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        of = outputs[:, :, :self.dimension_bottle]
        ob = outputs[:, :, self.dimension_bottle:]
        
        codes = []
        for i in range(0, outputs.size(1), 16):
            codes.append(torch.cat((of[:,i+16-1,:],ob[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    def __init__(self, dimension_bottle, embedding_dimension, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dimension_bottle*2+embedding_dimension, dim_pre, 1, batch_first=True)
        

        self.convolution_layer1 = torch.nn.Conv1d(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1,
                        bias=True)
        
        self.convolution_layer2 = torch.nn.Conv1d(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1,
                        bias=True)
        
        self.convolution_layer3 = torch.nn.Conv1d(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1,
                        bias=True)

        
        #self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = torch.nn.Linear(1024,80, bias=True)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        x = F.relu(self.convolution_layer1(x))
        x = F.relu(self.convolution_layer2(x))
        x = F.relu(self.convolution_layer3(x))

        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        outputs = self.linear_projection(outputs)

        return outputs   
    
 

class EncoderDecoder(nn.Module):
    
    def __init__(self, dimension_bottle, embedding_dimension, dim_pre):
        super(EncoderDecoder, self).__init__()
        
        self.encoder = Encoder(dimension_bottle, embedding_dimension)
        self.decoder = Decoder(dimension_bottle, embedding_dimension, dim_pre)
        

    def forward(self, x, speaker, c_trg):
                
        codes = self.encoder(x, speaker)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        
        return mel_outputs
