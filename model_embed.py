# model.py
# Music recommendation model for one vs. one classification

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

class UserMusicEmbedding(nn.Module):
    def __init__(self, user_embed_dim, music_embed_dim, n_user, n_music)