import math
import torch
import torch.nn as nn


def freq_positional_encoding(seq_len,emb_dim):
    """
    implement of positional encoding
    :param seq_len: the maximum sequence length
    :param emb_dim: the embedding dimension
    :return: positional encoding
    """
    pe=torch.zeros(seq_len,emb_dim)
    position=torch.arange(0,seq_len).unsqueeze(1) #(seq_len,1)
    div_term=torch.exp(torch.arange(0,emb_dim,2)*(-math.log(10000.0)/emb_dim)) 
    pe[:,0::2]=torch.sin(position*div_term)
    pe[:,1::2]=torch.cos(position*div_term)
    return pe.unsqueeze(0) #(1,seq_len,emb_dim)




class RotaryPositionalEncoding(nn.Module):
    """
    implement of rotary positional encoding
    """
    def __init__(self,config):
        super().__init__()
        self.seq_len=config.block_size
        self.emb_dim=config.emb_dim
        inv_freq=1.0/(config.theta**(torch.arange(0,config.emb_dim,2).float()/config.emb_dim))
        self.register_buffer("inv_freq",inv_freq)

    def forward(self,x):
        position=torch.arange(self.seq_len,device=x.device).type_as(self.inv_freq)
        freqs=torch.einsum("i,j->ij",position,self.inv_freq) ## Outer Prodct
        freqs=torch.cat((freqs,freqs),dim=-1)
        return freqs.cos(),freqs.sin()
    
def rotate_half(x):
    x1,x2=x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1) #[x_1,x_2,x_3,x_4]-->[-x_3,x_4,-_1,x_2] Not the same as the paper, but this may calculate a bit faster

# def rotate_interleaved(x):    
#     x1 = x[..., 0::2] 
#     x2 = x[..., 1::2]
#     return torch.stack((-x2, x1), dim=-1).flatten(-2)  

def apply_rotary_pos_emb(x,cos,sin):
    return (x*cos)+(rotate_half(x)*sin)