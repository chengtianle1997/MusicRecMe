# model.py
# Music recommendation model

import torch
import torch.nn as nn

class UserAttention(object):
    '''
    The user embedding model with attention on their playlists
    '''
    def __init__(self, music_embed_dim, embed_dim_list, dropout=0.0):
        '''
        music_embed_dim: dimension of music embedding
        embed_dim_list []: list of [genre, meta, audio, lyric] dimension
        dropout: dropout rate
        '''
        pass