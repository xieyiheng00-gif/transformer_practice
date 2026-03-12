import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math

class LayerNorm(nn.Module):
    """
    layer norm, could choose whether or not to have bias,
    it can reduce computation without bias but keep the quality of the model
    """
    def __init__(self,emb_dim,bias:bool):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(emb_dim))
        self.bias=nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self):
        return F.layer_norm(input,self.weight.shape,self.weight,self.bias,1e-6)



class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        """
        Validate that the embedding dimension is divided by head dimension
        calculate the head dimension.calculate the attention.
        config is a class for multiple data, like embedding dimension,number_head,droupout,bias:bool
        """
        super().__init__()
        assert config.emb_dim % config.n_head ==0,\
              f"Embeddind dimension({config.emb_dim}) must be exactly divisible by number of head({config.n_head})"
        self.n_head=config.n_head
        self.emb_dim=config.emb_dim
        self.dropout=config.dropout
        self.c_attn=nn.Linear(self.emb_dim,3*self.emb_dim,config.bias)
        self.c_proj=nn.Linear(self.emb_dim,self.emb_dim,config.bias)
        self.att_dropout=nn.Dropout(self.dropout)
        self.resid_dropout=nn.Dropout(self.dropout)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))   ##(casual mask)
        ##blcok_size is the maximum sequence length.   ### why we use bias is because of the convention

    def forward(self,x):
        b,t,c=x.size ###b:batch size  t:sequence length   c:embedding dimension
        d=self.emb_dim/self.n_head
        q,k,v=self.c_attn(x).split(self.emb_dim,dim=2)
        q=q.view(b,t,self.n_head,d).transpose(1,2)       ### view() do not change how tensor stored in real memory, just change how it looks like
        k=k.view(b,t,self.n_head,d).transpose(1,2)
        v=v.view(b,t,self.n_head,d).transpose(1,2)

        attention=(q @ k.transpose(-2,-1))/math.sqrt(d)
        attention=attention.maksed_fill(self.bias[:,:,:t,:t] ==0, float('-inf'))
        attention=F.softmax(attention,-1)
        attention=self.att_dropout(attention)
        attention=attention @ v
        attention=attention.transpose(1,2).contigous().view(b,t,c)
        attention=self.resid_dropout(self.c_proj(attention))
        return attention



class FeedForward(nn.Module):
    def __init__(self,config):
        """
        linear layers
        actiavte layers
        get hidden layer dimension, normally, it is 4X the embedding dimension.
        """
        super().__init()
        self.linear1=nn.linear(config.emb_dim,4*config.emb_dim,config.bias)
        self.gelu=nn.GELU()
        self.linear2=nn.Linear(4*config.emb_dim,config.emb_dim,config.bias)
        self.dropout=nn.Dropout(config.dropout)



    def forward(self,x):
        """
        implement of layer
        linear-->RELU(GULE)-->linear
        """
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.linear2(x)
        x=self.dropout(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1=LayerNorm(config.emb_dim,config.bias)
        self.att=MultiHeadAttention(config)
        self.ln2=LayerNorm(config.emb_dim,config.bias)
        self.mlp=FeedForward(config)
    



    def forward(self,x):
        x=x+self.att(self.ln1(x))
        x=x+self.mlp(self.ln2())
        return x

@dataclass
class GPTconfig():
    n_head: int=8
    emd_dim : int=128
    dropout :float =0.0
    bias : bool =False
    block_size :int = 1024



