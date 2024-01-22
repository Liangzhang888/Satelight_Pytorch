# -*- coding: utf-8 -*-
#
# model.py
#
# Created by LiangZhang on 2023/10/10
#
import torch 
import torch.nn as nn 
from torchinfo import summary 
from math import sqrt

class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_k, dim_v) -> None:
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1.0 / sqrt(dim_k)
    
    def forward(self, x):
        Q = self.q(x)
        K = self.k(x) 
        V = self.v(x) 
        
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1)))
        output = torch.bmm(atten, V)
        
        return output

class AttentionBlock(nn.Module):
    def __init__(self, n ,N=128, C=32) -> None:
        super().__init__()
        self.n = n
        
        self.MyBlock1 = nn.Sequential(
            Self_Attention(input_dim=C *n, dim_k = N // pow(4,n-1), dim_v=C*n), 
            nn.BatchNorm1d(N // pow(4,n-1)),
            nn.Dropout1d(p=0.2),
        )
        
        self.MyBlock2 = nn.Sequential(
            nn.Linear(C*n*2, C*(n+1)),
            nn.BatchNorm1d(N // pow(4,n-1)),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.2),    
        )
        self.MyBlock3 = nn.MaxPool2d(kernel_size=(3*N //pow(4,n)+1 , 3*C*(n+1) +1), padding=(0,0),stride=(1,1))
        self.pad = nn.ZeroPad2d((C*(n+3)//2 , C*(n+3)//2,0,0))
        
    def forward(self, x,):
        _local1_out = self.MyBlock1(x)
        _local2_in = torch.concat((x,_local1_out),dim=2)
        _local2_out = self.MyBlock2(_local2_in)
        _local3_in = torch.concat((_local2_out,self.pad(_local2_in)),dim=2)
        _local3_out = self.MyBlock3(_local3_in)
        
        return _local3_out

class Satelight(nn.Module):
    def __init__(self, c = 16) -> None:
        super(Satelight,self).__init__()
        
        self.Factorized = nn.Sequential(
            nn.Conv2d(1,c,kernel_size=(1,149),stride=1,padding=(0,74)),
            nn.Conv2d(c,2*c,kernel_size=(39,1),stride=1,padding=(0,0)),
        )
        self.Layer1 = nn.Sequential(
            nn.BatchNorm1d(2*c),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.2),
            nn.MaxPool2d(padding=(0,2),kernel_size=(1,50),stride=(1,2)),
        )
        self.Layer2 = AttentionBlock(n = 1)
        self.Layer3 = AttentionBlock(n = 2)
        self.Layer4 = AttentionBlock(n = 3) 
        
        
        self.classfier = nn.Sequential(
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(256,1),   
        )

    def forward(self, x):
        
        x = self.Factorized(x)
        x = x.squeeze(2)
        x = self.Layer1(x)
       
        x = self.Layer2(x.permute(0,2,1))
        # The target net is for (batch,128,32), But raw data's dimension is (batch,32,128).
        x = self.Layer3(x)
        x = self.Layer4(x)
        
        x = self.classfier(x)
        x = x.sigmoid()
        x = x.squeeze(1)
        
        return x 
    
if __name__ == "__main__":
    
    """
        todo SelfAttention Module test
    """
    # n =3
    # attention = Self_Attention(input_dim=32 *n, dim_k=128// pow(4,n-1), dim_v=32*n)
    # x = torch.randn(60, 128// pow(4,n-1), 32*n)
    # output = attention(x)
    # print(output.size())
    """
        todo AttentionBlock Module test
    """
    # n = 3
    # x = torch.randn(60, 128// pow(4,n-1), 32*n)
    # attentionblock = AttentionBlock(n=n) 
    # output = attentionblock(x)
    # print(output.size())  
    
    """
    todo Satelight Module test
    """
    # model = Satelight()
    # x = torch.randn(60,1,39,300)
    # output = model(x)
    # print(output.size())
    pass