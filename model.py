# model.py
# Music recommendation model

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1] # head_dim
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k) # [batch_size, num_heads, seq_length, seq_length]
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1) # [batch_size, num_heads, seq_length, seq_length]
    values = torch.matmul(attention, v)        # [batch_size, num_heads, seq_length, head_dim]
    return values, attention

class MultiheadAttention(nn.Module):
    # Multihead Attention
    def __init__(self, input_dim, embed_dim, num_heads):
        '''
        input_dim: input music embedding dimension
        embed_dim: embedding dimension for Q, K and V
        num_heads: number of multi-heads
        '''
        super().__init__()
        assert embed_dim % num_heads == 0, "[Multi-head Attention]: Embedding dimension must be 0 modulko number of heads!"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stock all weights matrices together for efficiency
        self.qkv_proj = nn.Linear(input_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        # reset all params
        self._reset_parameters()

    def _reset_parameters(self):
        # Original transformer initialization
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=True):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Seperate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, 3*head_dim]
        q, k, v = qkv.chunk(3, dim=-1) # q, k, v = [batch_size, num_heads, seq_length, head_dim]

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3) # [batch_size, seq_length, num_heads, head_dim]
        values = values.reshape(batch_size, seq_length, embed_dim)
        out = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class UserAttention(nn.Module):
    '''
    The user embedding model with attention on their playlists
    '''
    def __init__(self, music_embed_dim, music_embed_dim_list, embed_dim=128, dropout=0.1, num_heads=1, num_layer=2, re_embed=False):
        '''
        music_embed_dim: dimension of music embedding
        music_embed_dim_list []: list of [genre, meta, audio, lyric] dimension
        embed_dim: embedding dimension for Q, K, V projection in multi-head attention
        dropout: dropout rate
        '''
        super().__init__()

        # re-embed audio and lyric for a smaller attention dim
        if re_embed is True:
            pass
        
        self.music_embed_dim = music_embed_dim
        self.num_heads = num_heads

        # multi-head self-attention layer
        self.attn = MultiheadAttention(music_embed_dim, embed_dim, num_heads)

        # multi-layer
    
    def forward(self, x):
        
        # attention


