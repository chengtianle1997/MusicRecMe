# model.py
# Music recommendation model

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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

def generate_mask(len_list):
    max_len = max(len_list)
    mask_np = np.zeros((len(len_list), max_len, max_len)) # [batch_size, seq_length, seq_length]
    # mark valid positions
    for i in range(len(len_list)):
        mask_np[i, 0: len_list[i], 0: len_list[i]] = 1
    mask = torch.tensor(mask_np)
    return mask

def generate_out_mask(len_list):
    max_len = max(len_list)
    mask_np = np.zeros((len(len_list), max_len, 1))
    # mark valid positions
    for i in range(len(len_list)):
        mask_np[i, 0: len_list[i], 0] = 1
    mask = torch.tensor(mask_np)
    return mask

# input:
# y_mask = [batch_size, seq_length, 1]
def get_rmse_loss(mse_loss, pred, y, y_mask):
    pred = pred.reshape(pred.shape[0], 1, pred.shape[1]) # [batch_size, 1, embed_dim]
    pred = pred.repeat(1, y.shape[1], 1) # [batch_size, max_seq_len, embed_dim]
    y_mask = y_mask.repeat(1, 1, pred.shape[2])
    loss = mse_loss(pred, y)
    loss = (loss * y_mask).sum()
    valid_elements = y_mask.sum()
    loss = torch.sqrt(loss / valid_elements)
    return loss

class MusicRecommender(object):
    '''
    Recommend songs according to song dictionary and user embedding
    '''
    def __init__(self, dataset, device, mode='train'):
        self.device = device
        if mode == 'train':
            self.song_dict = dataset.train_song_dict
            self.song_mat = torch.tensor(dataset.train_song_mat).to(device)
        elif mode == 'test':
            self.song_dict = dataset.test_song_dict
            self.song_mat = torch.tensor(dataset.test_song_mat).to(device)
        # reshape: song_mat = [song_num, song_embed_dim, 1]
        # self.song_mat = self.song_mat.reshape(self.song_mat.shape[0], self.song_mat.shape[1], 1)
        self.cos_sim = nn.CosineSimilarity(dim=1)
        # reverse song dict to find track by index
        self.rev_song_dict = {v: k for k, v in self.song_dict.items()}
        # mse loss
        mse_loss = nn.MSELoss()
    
    # recommend songs according to cosine similarity (for multi-users at once)
    # user_embed[]: user embedding from UserAttention model [batch_size, user_embed_dim]
    # x_valid_tracks []: given track ids in x (not embeddings) [batch_size, seq_len]
    # y_valid_tracks []: ground truth track ids (not embeddings) [batch_size, seq_len]
    # return_songs: whether return recommended track ids or not
    # top_k: return top K recommended songs
    def recommend(self, user_embed, x_valid_tracks, y_valid_tracks, return_songs=False, top_k=100):
        top_10_recall_num = 0
        top_50_recall_num = 0
        top_100_recall_num = 0
        ground_truth_num = 0
        top_10_track_list = []
        # iterate through users
        for i in range(user_embed.shape[0]):
            # similarity = [song_num]
            similarity = self.cos_sim(self.song_mat, user_embed[i, :].reshape(1, user_embed.shape[1]))
            # mask out songs in x
            index_mask = torch.ones(similarity.shape[0]).to(self.device)
            for track in x_valid_tracks[i]:
                index_mask[self.song_dict[track]] = -1
            similarity = similarity * index_mask
            # top_k_index = [top_k]
            top_k_index = torch.topk(similarity.flatten(), top_k).indices
            # get track embedding
            top_k_embed = [self.song_mat[int(i)] for i in top_k_index]
            # get track ids
            top_k_track = [self.rev_song_dict[int(i)] for i in top_k_index]
            top_10_track_list.append(top_k_track)
            # get intersection of predicted track and groud truth tracks
            gt_set = set(y_valid_tracks[i])
            # top 10, 50, 100 recall
            top_10_inter = set(top_k_track[0:10]) & gt_set
            top_50_inter = set(top_k_track[0:50]) & gt_set
            top_100_inter = set(top_k_track[0:100]) & gt_set
            top_10_recall_num += len(top_10_inter)
            top_50_recall_num += len(top_50_inter)
            top_100_recall_num += len(top_100_inter)
            ground_truth_num += len(gt_set)
        
        recalls = [top_10_recall_num / ground_truth_num, top_50_recall_num / ground_truth_num, \
            top_100_recall_num / ground_truth_num]
        if return_songs is True:
            return top_10_track_list, recalls
        else:
            # calculate recall rate
            return recalls

        


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
        self.qkv_proj = nn.Linear(input_dim, embed_dim*3)
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
        values = values.reshape(batch_size, seq_length, embed_dim) # [batch_size, seq_length, embed_dim]
        out = self.o_proj(values) # [batch_size, seq_length, embed_dim]

        if return_attention:
            return out, attention
        else:
            return out

# Attention on playlists to generate user embeddings
class UserAttention(nn.Module):
    '''
    The user embedding model with attention on their playlists
    '''
    def __init__(self, music_embed_dim, music_embed_dim_list, embed_dim=None, dropout=0.1, num_heads=1, re_embed=False):
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

        if embed_dim is None:
            embed_dim = music_embed_dim
        
        self.music_embed_dim = music_embed_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # multi-head self-attention layer
        self.self_attn = MultiheadAttention(music_embed_dim, embed_dim, num_heads)

        # multi linear layers
        pass

        # normalize and dropout layer
        self.attn_out_norm =nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # input x: [batch_size, seq_length, music_embed_dim]
        # mask: [batch_size, seq_length, seq_length]

        # convert mask
        if mask is not None:
            mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2]) # [batch_size, 1, seq_length, seq_length]
            mask = mask.repeat(1, self.num_heads, 1, 1) # [batch_size, num_heads, seq_length, seq_length]
        
        # attention
        attn_out, attn_weights = self.self_attn(x, mask=mask) # [batch_size, seq_length, embed_dim]
        # residual connection
        x = x + self.dropout(attn_out)
        x = self.attn_out_norm(x) # [batch_size, seq_length, embed_dim]
        # sum among sequence
        x = x.mean(dim=1) # [batch_size, embed_dim]

        # linear layers
        pass
        
        return x

